# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2018, Xilun Chen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import os
from logging import getLogger
import random

import numpy as np
import scipy
import scipy.linalg
import torch
from torch.nn import functional as F
from torch import optim

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters, apply_mapping
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, embs, mappings, discriminators, params):
        """
        Initialize trainer script.
        """
        self.embs = embs
        self.vocabs = params.vocabs
        self.mappings = mappings
        self.discriminators = discriminators
        self.params = params
        self.dicos = {}

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(itertools.chain(*[m.parameters()
                for l,m in mappings.items() if l!=params.tgt_lang]), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(itertools.chain(*[d.parameters()
                for d in discriminators.values()]), **optim_params)
        else:
            assert discriminators is None
        if hasattr(params, 'mpsr_optimizer'):
            optim_fn, optim_params = get_optimizer(params.mpsr_optimizer)
            self.mpsr_optimizer = optim_fn(itertools.chain(*[m.parameters()
                for l,m in self.mappings.items() if l!=self.params.tgt_lang]), **optim_params)

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False
        self.decrease_mpsr_lr = False

    def get_dis_xy(self, lang1, lang2, volatile):
        """
        Get discriminator input batch / output target.
        Encode from lang1, decode to lang2 and then discriminate
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(map(len, self.vocabs.values()))
        src_ids = torch.LongTensor(bs).random_(len(self.vocabs[lang1]) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.vocabs[lang2]) if mf == 0 else mf)
        src_ids = src_ids.to(self.params.device)
        tgt_ids = tgt_ids.to(self.params.device)

        with torch.set_grad_enabled(not volatile):
            # get word embeddings
            src_emb = self.embs[lang1](src_ids).detach()
            tgt_emb = self.embs[lang2](tgt_ids).detach()
            # map
            src_emb = self.mappings[lang1](src_emb)
            # decode
            src_emb = F.linear(src_emb, self.mappings[lang2].weight.t())

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        # 0 indicates real (lang2) samples
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = y.to(self.params.device)

        return x, y

    def get_mpsr_xy(self, lang1, lang2, volatile):
        """
        Get input batch / output target for MPSR.
        """
        # select random word IDs
        bs = self.params.batch_size
        dico = self.dicos[(lang1, lang2)]
        indices = torch.from_numpy(np.random.randint(0, len(dico), bs)).to(self.params.device)
        dico = dico.index_select(0, indices)
        src_ids = dico[:, 0].to(self.params.device)
        tgt_ids = dico[:, 1].to(self.params.device)

        with torch.set_grad_enabled(not volatile):
            # get word embeddings
            src_emb = self.embs[lang1](src_ids).detach()
            tgt_emb = self.embs[lang2](tgt_ids).detach()
            # map
            src_emb = self.mappings[lang1](src_emb)
            # decode
            src_emb = F.linear(src_emb, self.mappings[lang2].weight.t())

        return src_emb, tgt_emb

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        for disc in self.discriminators.values():
            disc.train()

        # loss
        loss = 0
        # for each target language
        for lang2 in self.params.all_langs:
            # random select a source language
            lang1 = random.choice(self.params.all_langs)

            x, y = self.get_dis_xy(lang1, lang2, volatile=True)
            preds = self.discriminators[lang2](x.detach())
            loss += F.binary_cross_entropy(preds, y)

        # check NaN
        if (loss != loss).any():
            logger.error("NaN detected (discriminator)")
            exit()
        stats['DIS_COSTS'].append(loss.item())

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        for d in self.discriminators:
            clip_parameters(d, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        for disc in self.discriminators.values():
            disc.eval()

        # loss
        loss = 0
        for lang1 in self.params.all_langs:
            lang2 = random.choice(self.params.all_langs)

            x, y = self.get_dis_xy(lang1, lang2, volatile=False)
            preds = self.discriminators[lang2](x)
            loss += F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return len(self.params.all_langs) * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        # load dicos for all lang pairs
        for i, lang1 in enumerate(self.params.all_langs):
            for j, lang2 in enumerate(self.params.all_langs):
                if lang1 == lang2:
                    idx = torch.arange(self.params.dico_max_rank).long().view(self.params.dico_max_rank, 1)
                    self.dicos[(lang1, lang2)] = torch.cat([idx, idx], dim=1).to(self.params.device)
                else:
                    word2id1 = self.vocabs[lang1].word2id
                    word2id2 = self.vocabs[lang2].word2id

                    # identical character strings
                    if dico_train == "identical_char":
                        self.dicos[(lang1, lang2)] = load_identical_char_dico(word2id1, word2id2)
                    # use one of the provided dictionary
                    elif dico_train == "default":
                        filename = '%s-%s.0-5000.txt' % (lang1, lang2)
                        self.dicos[(lang1, lang2)] = load_dictionary(
                            os.path.join(DIC_EVAL_PATH, filename),
                            word2id1, word2id2
                        )
                    # TODO dictionary provided by the user
                    else:
                        # self.dicos[(lang1, lang2)] = load_dictionary(dico_train, word2id1, word2id2)
                        raise NotImplemented(dico_train)
                    self.dicos[(lang1, lang2)] = self.dicos[(lang1, lang2)].to(self.params.device)

    def build_dictionary(self):
        """
        Build dictionaries from aligned embeddings.
        """
        # build dicos for all lang pairs
        for i, lang1 in enumerate(self.params.all_langs):
            for j, lang2 in enumerate(self.params.all_langs):
                if i < j:
                    src_emb = self.embs[lang1].weight
                    src_emb = apply_mapping(self.mappings[lang1], src_emb).detach()
                    tgt_emb = self.embs[lang2].weight
                    tgt_emb = apply_mapping(self.mappings[lang2], tgt_emb).detach()
                    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
                    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
                    self.dicos[(lang1, lang2)] = build_dictionary(src_emb, tgt_emb, self.params)
                elif i > j:
                    self.dicos[(lang1, lang2)] = self.dicos[(lang2, lang1)][:, [1,0]]
                else:
                    idx = torch.arange(self.params.dico_max_rank).long().view(self.params.dico_max_rank, 1)
                    self.dicos[(lang1, lang2)] = torch.cat([idx, idx], dim=1).to(self.params.device)

    def mpsr_step(self, stats):
        # loss
        loss = 0
        for lang1 in self.params.all_langs:
            lang2 = random.choice(self.params.all_langs)

            x, y = self.get_mpsr_xy(lang1, lang2, volatile=False)
            loss += F.mse_loss(x, y)
        # check NaN
        if (loss != loss).any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        stats['MPSR_COSTS'].append(loss.item())
        # optim
        self.mpsr_optimizer.zero_grad()
        loss.backward()
        self.mpsr_optimizer.step()

        if self.params.mpsr_orthogonalize:
            self.orthogonalize()

        return len(self.params.all_langs) * self.params.batch_size

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            for mapping in self.mappings.values():
                W = mapping.weight.detach()
                beta = self.params.map_beta
                W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def update_mpsr_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.mpsr_optimizer:
            return
        old_lr = self.mpsr_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.mpsr_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_mpsr_lr:
                    old_lr = self.mpsr_optimizer.param_groups[0]['lr']
                    self.mpsr_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.mpsr_optimizer.param_groups[0]['lr']))
                self.decrease_mpsr_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            tgt_lang = self.params.tgt_lang
            for src_lang in self.params.src_langs:
                W = self.mappings[src_lang].weight.detach().cpu().numpy()
                path = os.path.join(self.params.exp_path,
                                    f'best_mapping_{src_lang}2{tgt_lang}.t7')
                logger.info(f'* Saving the {src_lang} to {tgt_lang} mapping to %s ...' % path)
                torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        tgt_lang = self.params.tgt_lang
        for src_lang in self.params.src_langs:
            path = os.path.join(self.params.exp_path,
                                f'best_mapping_{src_lang}2{tgt_lang}.t7')
            logger.info(f'* Reloading the best {src_lang} to {tgt_lang} model from {path} ...')
            # reload the model
            assert os.path.isfile(path)
            to_reload = torch.from_numpy(torch.load(path))
            W = self.mappings[src_lang].weight.detach()
            assert to_reload.size() == W.size()
            W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params
        # load all embeddings
        logger.info("Reloading embeddings for mapping ...")
        params.vocabs[params.tgt_lang], tgt_emb = load_embeddings(params, params.tgt_lang,
                params.tgt_emb, full_vocab=True)
        normalize_embeddings(tgt_emb, params.normalize_embeddings,
                mean=params.lang_mean[params.tgt_lang])
        # export target embeddings
        export_embeddings(tgt_emb, self.params.tgt_lang, self.params)
        # export all source embeddings
        for i, src_lang in enumerate(self.params.src_langs):
            params.vocabs[src_lang], src_emb = load_embeddings(params, src_lang,
                    params.src_embs[i], full_vocab=True)
            logger.info(f"Map {src_lang} embeddings to the target space ...")
            src_emb = apply_mapping(self.mappings[src_lang], src_emb)
            export_embeddings(src_emb, src_lang, self.params)
