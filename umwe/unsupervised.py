# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2018, Xilun Chen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'
# default path to embeddings embeddings if not otherwise specified
EMB_DIR = 'data/fasttext-vectors/'


# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
# parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--device", type=str, default="cuda", help="Run on GPU or CPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_langs", type=str, nargs='+', default=['de', 'es', 'fr', 'it', 'pt'], help="Source languages")
parser.add_argument("--tgt_lang", type=str, default='en', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# MPSR parameters
parser.add_argument("--mpsr_optimizer", type=str, default="adam", help="Multilingual Pseudo-Supervised Refinement optimizer")
parser.add_argument("--mpsr_orthogonalize", type=bool_flag, default=True, help="During MPSR, whether to perform orthogonalization")
parser.add_argument("--mpsr_n_steps", type=int, default=30000, help="Number of optimization steps for MPSR")
# dictionary creation parameters (for refinement)
# default uses .5000-6500.txt; train uses .0-5000.txt; all uses .txt
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--semeval_ignore_oov", type=bool_flag, default=True, help="Whether to ignore OOV in SEMEVAL evaluation (the original authors used True)")
# reload pre-trained embeddings
parser.add_argument("--src_embs", type=str, nargs='+', default=[], help="Reload source embeddings (should be in the same order as in src_langs)")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# post-processing options
params.src_N = len(params.src_langs)
params.all_langs = params.src_langs + [params.tgt_lang]
# load default embeddings if no embeddings specified
if len(params.src_embs) == 0:
    params.src_embs = []
    for lang in params.src_langs:
        params.src_embs.append(os.path.join(EMB_DIR, f'wiki.{lang}.vec'))
if len(params.tgt_emb) == 0:
    params.tgt_emb = os.path.join(EMB_DIR, f'wiki.{params.tgt_lang}.vec')

# check parameters
assert not params.device.lower().startswith('cuda') or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert all([os.path.isfile(emb) for emb in params.src_embs])
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build model / trainer / evaluator
logger = initialize_exp(params)
# N+1 embeddings, N mappings , N+1 discriminators
embs, mappings, discriminators = build_model(params, True)
trainer = Trainer(embs, mappings, discriminators, params)
evaluator = Evaluator(trainer)


"""
Learning loop for Multilingual Adversarial Training
"""
if params.adversarial:
    logger.info('----> MULTILINGUAL ADVERSARIAL TRAINING <----\n\n')

    # training loop
    for n_epoch in range(params.n_epochs):

        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': []}

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminator training
            for _ in range(params.dis_steps):
                trainer.dis_step(stats)

            # mapping training (discriminator fooling)
            n_words_proc += trainer.mapping_step(stats)

            # log stats
            if n_iter % 500 == 0:
                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

                # reset
                tic = time.time()
                n_words_proc = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        evaluator.all_eval(to_log)
        evaluator.eval_all_dis(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of epoch %i.\n\n' % n_epoch)

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log, VALIDATION_METRIC)
        if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break


"""
Learning loop for Multilingual Pseudo-Supervised Refinement
"""
if params.n_refinement > 0:
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('----> MULTILINGUAL PSEUDO-SUPERVISED REFINEMENT <----\n\n')
    trainer.reload_best()

    # training loop
    for n_epoch in range(params.n_refinement):

        logger.info('Starting refinement iteration %i...' % n_epoch)

        # build a dictionary from aligned embeddings
        trainer.build_dictionary()

        # optimize MPSR
        tic = time.time()
        n_words_mpsr = 0
        stats = {'MPSR_COSTS': []}
        for n_iter in range(params.mpsr_n_steps):
            # mpsr training step
            n_words_mpsr += trainer.mpsr_step(stats)
            # log stats
            if n_iter % 500 == 0:
                stats_str = [('MPSR_COSTS', 'MPSR loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_mpsr / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                # reset
                tic = time.time()
                n_words_mpsr = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # embeddings evaluation
        to_log = OrderedDict({'n_mpsr_epoch': n_epoch})
        evaluator.all_eval(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of refinement iteration %i.\n\n' % n_epoch)

        # update the learning rate (effective only if using SGD for MPSR)
        trainer.update_mpsr_lr(to_log, VALIDATION_METRIC)


# export embeddings
if params.export:
    trainer.reload_best()
    trainer.export()
