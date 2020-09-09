# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2018, Xilun Chen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor as torch_tensor
from torch.nn import functional as F

from . import get_wordsim_scores, get_crosslingual_wordsim_scores, get_wordanalogy_scores
from . import get_word_translation_accuracy
from . import load_europarl_data, get_sent_translation_accuracy
from ..dico_builder import get_candidates, build_dictionary
from src.utils import get_idf, apply_mapping


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.embs = trainer.embs
        self.vocabs = trainer.vocabs
        self.mappings = trainer.mappings
        self.discriminators = trainer.discriminators
        self.params = trainer.params

    def monolingual_wordsim(self, to_log):
        """
        Evaluation on monolingual word similarity.
        """
        ws_monolingual_scores = {}
        for lang in self.params.all_langs:
            ws_scores = get_wordsim_scores(
                lang, self.vocabs[lang].word2id,
                apply_mapping(self.mappings[lang], self.embs[lang].weight.detach()).cpu().numpy()
            )
            if ws_scores is None:
                continue
            ws_monolingual_scores[lang] = np.mean(list(ws_scores.values()))
            logger.info("Monolingual %s word similarity score average: %.5f" % (lang, ws_monolingual_scores[lang]))
            to_log[f'{lang}_ws_monolingual_scores'] = ws_monolingual_scores[lang]
            to_log.update({f'{lang}_{k}': v for k, v in ws_scores.items()})

        if len(ws_monolingual_scores) == 0:
            return
        avg_ws_monolingual_score = sum(ws_monolingual_scores.values()) / len(ws_monolingual_scores) 
        logger.info("Monolingual word similarity score average: %.5f" % avg_ws_monolingual_score)
        to_log['ws_monolingual_scores'] = avg_ws_monolingual_score

    def monolingual_wordanalogy(self, to_log):
        """
        Evaluation on monolingual word analogy.
        """
        analogy_monolingual_scores = {}
        for lang in self.params.all_langs:
            analogy_scores = get_wordanalogy_scores(
                lang, self.vocabs[lang].word2id,
                apply_mapping(self.mappings[lang], self.embs[lang].weight.detach()).cpu().numpy()
            )
            if analogy_scores is None:
                continue
            analogy_monolingual_scores[lang] = np.mean(list(analogy_scores.values()))
            logger.info("Monolingual %s word analogy score average: %.5f" % (lang, analogy_monolingual_scores))
            to_log[f'{lang}_analogy_monolingual_scores'] = analogy_monolingual_scores[lang]
        if len(analogy_monolingual_scores) == 0:
            return
        avg_analogy_monolingual_score = sum(analogy_monolingual_scores.values()) / len(analogy_monolingual_scores) 
        logger.info("Monolingual word analogy score average: %.5f" % avg_analogy_monolingual_score)
        to_log['analogy_monolingual_scores'] = avg_analogy_monolingual_score

    def crosslingual_wordsim(self, to_log, src_lang=None, tgt_lang=None):
        """
        Evaluation on cross-lingual word similarity.
        If src_lang and tgt_lang are not specified, evaluate all src_langs to tgt_lang
        """
        # evaluate all src langs to tgt_lang by default
        if src_lang is None and tgt_lang is None:
            ws_crosslingual_scores = []
            tgt_lang = self.params.tgt_lang
            tgt_emb = self.embs[tgt_lang].weight.detach().cpu().numpy()
            for src_lang in self.params.src_langs:
                src_emb = apply_mapping(self.mappings[src_lang],
                        self.embs[src_lang].weight.detach()).cpu().numpy()
                # cross-lingual wordsim evaluation
                ws_scores = get_crosslingual_wordsim_scores(
                    src_lang, self.vocabs[src_lang].word2id, src_emb,
                    tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
                    ignore_oov = self.params.semeval_ignore_oov
                )
                if ws_scores is None:
                    continue
                ws_crosslingual_score = np.mean(list(ws_scores.values()))
                ws_crosslingual_scores.append(ws_crosslingual_score)
                logger.info("%s-%s cross-lingual word similarity score: %.5f" % (src_lang, tgt_lang, ws_crosslingual_score))
                to_log[f'{src_lang}_{tgt_lang}_ws_crosslingual_scores'] = ws_crosslingual_score
                to_log.update({f'{src_lang}_{tgt_lang}_{k}': v for k, v in ws_scores.items()})

            avg_ws_crosslingual_score = np.mean(ws_crosslingual_scores)
            logger.info("Cross-lingual word similarity score average: %.5f" % avg_ws_crosslingual_score)
            to_log['ws_crosslingual_scores'] = avg_ws_crosslingual_score
        else:
            # only evaluate src_lang to tgt_lang; bridge as necessary
            assert src_lang is not None and tgt_lang is not None
            # encode src
            src_emb = apply_mapping(self.mappings[src_lang],
                    self.embs[src_lang].weight).cpu().numpy()
            # encode tgt
            tgt_emb = apply_mapping(self.mappings[tgt_lang],
                    self.embs[tgt_lang].weight).cpu().numpy()
            # cross-lingual wordsim evaluation
            ws_scores = get_crosslingual_wordsim_scores(
                src_lang, self.vocabs[src_lang].word2id, src_emb,
                tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
            )
            if ws_scores is None:
                return
            ws_crosslingual_score = np.mean(list(ws_scores.values()))
            logger.info("%s-%s cross-lingual word similarity score: %.5f" % (src_lang, tgt_lang, ws_crosslingual_score))
            to_log[f'{src_lang}_{tgt_lang}_ws_crosslingual_scores'] = ws_crosslingual_score
            to_log.update({f'{src_lang}_{tgt_lang}_{k}': v for k, v in ws_scores.items()})

    def word_translation(self, to_log, src_lang=None, tgt_lang=None):
        """
        Evaluation on word translation.
        If src_lang and tgt_lang are not specified, evaluate all src_langs to tgt_lang
        """
        # evaluate all src langs to tgt_lang by default
        if src_lang is None and tgt_lang is None:
            wt_precisions = []
            tgt_lang = self.params.tgt_lang
            tgt_emb = self.embs[tgt_lang].weight.detach()
            for src_lang in self.params.src_langs:
                # mapped word embeddings
                src_emb = apply_mapping(self.mappings[src_lang],
                        self.embs[src_lang].weight.detach())

                for method in ['nn', 'csls_knn_10']:
                    results = get_word_translation_accuracy(
                        src_lang, self.vocabs[src_lang].word2id, src_emb,
                        tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
                        method=method, dico_eval=self.params.dico_eval
                    )
                    if results is None:
                        continue
                    to_log.update([('%s-%s_%s-%s' % (src_lang, tgt_lang, k, method), v) for k, v in results])
                    if method == 'csls_knn_10':
                        for k, v in results:
                            if k == 'precision_at_1':
                                wt_precisions.append(v)
            to_log['precision_at_1-csls_knn_10'] = np.mean(wt_precisions)
            logger.info("word translation precision@1: %.5f" % (np.mean(wt_precisions)))
        else:
            # only evaluate src_lang to tgt_lang; bridge as necessary
            assert src_lang is not None and tgt_lang is not None
            # encode src
            src_emb = apply_mapping(self.mappings[src_lang],
                    self.embs[src_lang].weight).cpu()
            # encode tgt
            tgt_emb = apply_mapping(self.mappings[tgt_lang],
                    self.embs[tgt_lang].weight).cpu()
            for method in ['nn', 'csls_knn_10']:
                results = get_word_translation_accuracy(
                    src_lang, self.vocabs[src_lang].word2id, src_emb,
                    tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
                    method=method, dico_eval=self.params.dico_eval
                )
                if results is None:
                    continue
                to_log.update([('%s-%s_%s-%s' % (src_lang, tgt_lang, k, method), v) for k, v in results])

    def sent_translation(self, to_log, src_lang=None, tgt_lang=None):
        """
        Evaluation on sentence translation.
        If src_lang and tgt_lang are not specified, evaluate all src_langs to tgt_lang
        Only available on Europarl, for en - {de, es, fr, it} language pairs.
        """
        # parameters
        n_keys = 200000
        n_queries = 2000
        n_idf = 300000

        # load europarl data
        if not hasattr(self, 'europarl_data'):
            self.europarl_data = {}

        # evaluate all src langs to tgt_lang by default
        if src_lang is None and tgt_lang is None:
            tgt_lang = self.params.tgt_lang
            for src_lang in self.params.src_langs:
                lang_pair = (src_lang, tgt_lang)
                # load europarl data
                if lang_pair not in self.europarl_data:
                    self.europarl_data[lang_pair] = load_europarl_data(
                        src_lang, tgt_lang, n_max=(n_keys + 2 * n_idf)
                    )
                # if no Europarl data for this language pair
                if not self.europarl_data or lang_pair not in self.europarl_data \
                        or self.europarl_data[lang_pair] is None:
                    logger.info(f'Europarl data not found for {src_lang}-{tgt_lang}.')
                    continue

                # mapped word embeddings
                src_emb = apply_mapping(self.mappings[src_lang],
                        self.embs[src_lang].weight)
                tgt_emb = self.embs[tgt_lang].weight

                # get idf weights
                idf = get_idf(self.europarl_data[lang_pair], src_lang, tgt_lang, n_idf=n_idf)

                for method in ['nn', 'csls_knn_10']:
                    # source <- target sentence translation
                    results = get_sent_translation_accuracy(
                        self.europarl_data[lang_pair],
                        src_lang, self.vocabs[src_lang].word2id, src_emb,
                        tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
                        n_keys=n_keys, n_queries=n_queries,
                        method=method, idf=idf
                    )
                    to_log.update([('%s_to_%s_%s-%s' % (tgt_lang, src_lang, k, method), v) for k, v in results])
                    # target <- source sentence translation
                    results = get_sent_translation_accuracy(
                        self.europarl_data[lang_pair],
                        tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
                        src_lang, self.vocabs[src_lang].word2id, src_emb,
                        n_keys=n_keys, n_queries=n_queries,
                        method=method, idf=idf
                    )
                    to_log.update([('%s_to_%s_%s-%s' % (src_lang, tgt_lang, k, method), v) for k, v in results])
        else:
            # only evaluate src_lang to tgt_lang; bridge as necessary
            assert src_lang is not None and tgt_lang is not None
            lang_pair = (src_lang, tgt_lang)
            # load europarl data
            if lang_pair not in self.europarl_data:
                self.europarl_data[lang_pair] = load_europarl_data(
                    src_lang, tgt_lang, n_max=(n_keys + 2 * n_idf)
                )
            # if no Europarl data for this language pair
            if not self.europarl_data or lang_pair not in self.europarl_data \
                    or self.europarl_data[lang_pair] is None:
                logger.info(f'Europarl data not found for {src_lang}-{tgt_lang}.')
                return
            # encode src
            src_emb = apply_mapping(self.mappings[src_lang],
                    self.embs[src_lang].weight)
            # encode tgt
            tgt_emb = apply_mapping(self.mappings[tgt_lang],
                    self.embs[tgt_lang].weight)
            # get idf weights
            idf = get_idf(self.europarl_data[lang_pair], src_lang, tgt_lang, n_idf=n_idf)

            for method in ['nn', 'csls_knn_10']:
                # source <- target sentence translation
                results = get_sent_translation_accuracy(
                    self.europarl_data[lang_pair],
                    src_lang, self.vocabs[src_lang].word2id, src_emb,
                    tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
                    n_keys=n_keys, n_queries=n_queries,
                    method=method, idf=idf
                )
                to_log.update([('%s_to_%s_%s-%s' % (tgt_lang, src_lang, k, method), v) for k, v in results])
                # target <- source sentence translation
                results = get_sent_translation_accuracy(
                    self.europarl_data[lang_pair],
                    tgt_lang, self.vocabs[tgt_lang].word2id, tgt_emb,
                    src_lang, self.vocabs[src_lang].word2id, src_emb,
                    n_keys=n_keys, n_queries=n_queries,
                    method=method, idf=idf
                )
                to_log.update([('%s_to_%s_%s-%s' % (src_lang, tgt_lang, k, method), v) for k, v in results])

    def dist_mean_cosine(self, to_log):
        """
        Mean-cosine model selection criterion.
        """
        # all pair refine
        mean_cosines = []
        for i, src_lang in enumerate(self.params.src_langs):
            # mapped word embeddings
            src_emb = apply_mapping(self.mappings[src_lang],
                    self.embs[src_lang].weight)
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            for j in range(i+1, len(self.params.all_langs)):
                tgt_lang = self.params.all_langs[j]
                tgt_emb = apply_mapping(self.mappings[tgt_lang],
                        self.embs[tgt_lang].weight)
                tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
                # build dictionary
                # for dico_method in ['nn', 'csls_knn_10']:
                for dico_method in ['csls_knn_10']:
                    dico_build = 'S2T'
                    dico_max_size = 10000
                    # temp params / dictionary generation
                    _params = deepcopy(self.params)
                    _params.dico_method = dico_method
                    _params.dico_build = dico_build
                    _params.dico_threshold = 0
                    _params.dico_max_rank = 10000
                    _params.dico_min_size = 0
                    _params.dico_max_size = dico_max_size
                    s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
                    t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
                    dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
                    # mean cosine
                    if dico is None:
                        mean_cosine = -1e9
                    else:
                        mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                    mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
                    logger.info("%s-%s: Mean cosine (%s method, %s build, %i max size): %.5f"
                                % (src_lang, tgt_lang, dico_method, _params.dico_build, dico_max_size, mean_cosine))
                    to_log['%s-%s-mean_cosine-%s-%s-%i' % (src_lang, tgt_lang, dico_method, _params.dico_build, dico_max_size)] = mean_cosine
                    mean_cosines.append(mean_cosine)
        # average cosine across lang pairs
        to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = np.mean(list(mean_cosines))

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        self.monolingual_wordsim(to_log)
        self.crosslingual_wordsim(to_log)
        self.word_translation(to_log)
        self.sent_translation(to_log)
        self.dist_mean_cosine(to_log)

    def eval_discriminator(self, to_log, lang, real_preds, fake_preds):
        real_pred = np.mean(real_preds)
        fake_pred = np.mean(fake_preds)
        logger.info("%s Discriminator average real / fake predictions: %.5f / %.5f"
                    % (lang, real_pred, fake_pred))

        real_accu = np.mean([x < 0.5 for x in real_preds])
        fake_accu = np.mean([x >= 0.5 for x in fake_preds])
        dis_accu = ((fake_accu * len(fake_preds) + real_accu * len(real_preds)) /
                    (len(real_preds) + len(fake_preds)))
        logger.info("%s Discriminator real / fake / global accuracy: %.5f / %.5f / %.5f"
                    % (lang, real_accu, fake_accu, dis_accu))

        to_log[f'{lang}_dis_accu'] = dis_accu
        to_log[f'{lang}_dis_fake_pred'] = fake_pred
        to_log[f'{lang}_dis_real_pred'] = real_pred
        return dis_accu

    def eval_all_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        for disc in self.discriminators.values():
            disc.eval()

        # for src lang discriminator, eval tgt->src
        dis_accus = {}
        tgt_lang = self.params.tgt_lang
        for src_lang in self.params.src_langs:
            real_preds = []
            fake_preds = []
            for i in range(0, self.embs[src_lang].num_embeddings, bs):
                with torch.no_grad():
                    emb = self.embs[src_lang].weight[i:i + bs].detach()
                    preds = self.discriminators[src_lang](emb)
                real_preds.extend(preds.cpu().tolist())
            for i in range(0, self.embs[tgt_lang].num_embeddings, bs):
                with torch.no_grad():
                    emb = self.embs[tgt_lang].weight[i:i + bs].detach()
                    emb = F.linear(emb, self.mappings[src_lang].weight.t())
                    preds = self.discriminators[src_lang](emb)
                fake_preds.extend(preds.cpu().tolist())
            dis_accus[src_lang] = self.eval_discriminator(to_log, src_lang, real_preds, fake_preds)

        # for tgt lang, random sample fake examples from all src langs
        real_preds = []
        fake_preds = []
        for i in range(0, self.embs[tgt_lang].num_embeddings, bs):
            with torch.no_grad():
                emb = self.embs[tgt_lang].weight[i:i + bs].detach()
                preds = self.discriminators[src_lang](emb)
            real_preds.extend(preds.cpu().tolist())
        for src_lang in self.params.src_langs:
            # sub-sample
            for i in range(0, self.embs[src_lang].num_embeddings // self.params.src_N, bs):
                with torch.no_grad():
                    emb = self.embs[src_lang].weight[i:i + bs].detach()
                    preds = self.discriminators[src_lang](self.mappings[src_lang](emb))
                fake_preds.extend(preds.cpu().tolist())
        dis_accus[tgt_lang] = self.eval_discriminator(to_log, tgt_lang, real_preds, fake_preds)

        avg_dis_accu = np.mean(list(dis_accus.values()))
        to_log[f'dis_accu'] = avg_dis_accu
