# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2018, Xilun Chen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate.py --src_langs de fr es it pt --tgt_lang en --eval_pairs all

import os
import argparse
from collections import OrderedDict

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

# default path to embeddings embeddings if not otherwise specified
EMB_DIR = 'data/fasttext-vectors/'


# main
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
# parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--device", type=str, default="cuda", help="Run on GPU or CPU")
# data
parser.add_argument("--src_langs", type=str, nargs='+', default=[], help="Source languages")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
# evaluation
parser.add_argument("--eval_pairs", type=str, nargs='+', default=[], help="Language pairs to evaluate. e.g. ['en-de', 'de-fr']")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dict_suffix", type=str, default="5000-6500.txt", help="suffix to use for word translation (0-5000.txt or 5000-6500.txt or txt)")
parser.add_argument("--semeval_ignore_oov", type=bool_flag, default=True, help="Whether to ignore OOV in SEMEVAL evaluation (the original authors used True)")
# reload pre-trained embeddings
parser.add_argument("--src_embs", type=str, nargs='+', default=[], help="Reload source embeddings (should be in the same order as in src_langs)")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# post-processing options
params.src_N = len(params.src_langs)
params.all_langs = params.src_langs + [params.tgt_lang]
# load default embeddings
if len(params.src_embs) == 0:
    params.src_embs = []
    for lang in params.src_langs:
        params.src_embs.append(os.path.join(EMB_DIR, f'wiki.{lang}.vec'))
if len(params.tgt_emb) == 0:
    params.tgt_emb = os.path.join(EMB_DIR, f'wiki.{params.tgt_lang}.vec')
# expand 'all' in eval_pairs
if 'all' in params.eval_pairs:
    params.eval_pairs = []
    for lang1 in params.all_langs:
        for lang2 in params.all_langs:
            if lang1 != lang2:
                params.eval_pairs.append(f'{lang1}-{lang2}')

# check parameters
assert len(params.src_langs) > 0, "source language undefined"
assert all([os.path.isfile(emb) for emb in params.src_embs])
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

# build logger / model / trainer / evaluator
logger = initialize_exp(params, dump_params=False, log_name='evaluate.log')
embs, mappings, _ = build_model(params, False)
trainer = Trainer(embs, mappings, None, params)
trainer.reload_best()
evaluator = Evaluator(trainer)

# run evaluations
to_log = OrderedDict({'n_iter': 0})
all_wt = []
evaluator.monolingual_wordsim(to_log)
for eval_pair in params.eval_pairs:
    parts = eval_pair.split('-')
    assert len(parts) == 2, 'Invalid format for evaluation pairs.'
    src_lang, tgt_lang = parts[0], parts[1]
    logger.info(f'Evaluating language pair: {src_lang} - {tgt_lang}')
    evaluator.crosslingual_wordsim(to_log, src_lang=src_lang, tgt_lang=tgt_lang)
    evaluator.word_translation(to_log, src_lang=src_lang, tgt_lang=tgt_lang)
    all_wt.append(to_log[f'{src_lang}-{tgt_lang}_precision_at_1-csls_knn_10'])
    evaluator.sent_translation(to_log, src_lang=src_lang, tgt_lang=tgt_lang)

logger.info(f"Overall Word Translation Precision@1 over {len(all_wt)} language pairs: {sum(all_wt)/len(all_wt)}")
