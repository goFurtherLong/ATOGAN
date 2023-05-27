# Copyright (c) 2017-present, Facebook, Inc.
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

#from src.classifier import Classifier
from src.utils import bool_flag, initialize_exp, normalize_embeddings
from src.models import build_model
from src.trainer import Trainer
from src.evaluation.evaluator import Evaluator


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'
VALIDATION_METRIC_S2T = 'best-acc-S2T-precision_at_1'
# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="best-result-refine", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")#pso_en_ru_lr=0.2_psothreshold=2_test5
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='ru', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=1, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")

parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_steps_2", type=int, default=3, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")

# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=100000, help="Iterations per epoch")

parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--score_size", type=int, default=1024, help="score")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")

parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")

parser.add_argument('--lr', type=float, default=0.2, help='initial learning rate')
parser.add_argument('--dis_lr', type=float, default=0.05, help='initial learning rate')
parser.add_argument("--d_cycle_weight", type=float, default=1)
parser.add_argument("--d_weight", type=float, default=0.5)
parser.add_argument("--mono_weight", type=float, default=0.4)
parser.add_argument("--cycle_weight", type=float, default=5.0)

#EVOLUTIONARY GAN 
parser.add_argument("--candi_num", type=int,default=1,help="the num of survived children")
parser.add_argument("--lambda_f", type=float, default=0.4, help = "factor of Fd")

#parser.add_argument("--lambda_w", type=float, default=0, help = "factor of Fd")


parser.add_argument("--uns_f", type=float, default=0.5, help = "factor of unsupervised")

# training refinement
parser.add_argument("--n_refinement", type=int, default=10, help="Number of refinement iterations (0 to disable the refinement procedure)")
parser.add_argument("--save_refine", type=bool, default=False, help="whether to save the parameters after refinement")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=20000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="../FMGAN/data/wiki.en.vec", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="../FMGAN/data/wiki.ru.vec", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

#pso settings
parser.add_argument("--c1", type=float, default=1.8, help='pso parameter c1 ')
parser.add_argument("--c2", type=float, default=1.8, help='pso parameter c2 ')
parser.add_argument("--W", type=float, default=0.729, help='pso parameter W ')
parser.add_argument("--pso_epoch_size", type=int, default=1024, help="Iterations per epoch")
parser.add_argument('--particle_dim', type=int, default=5, help='the dimension of each particle')
parser.add_argument('--particle_num', type=int, default=20, help='the size of particle swarm')
parser.add_argument('--iterations', type=int, default=50, help='the maximum number of iterations')
parser.add_argument('--use_pso_threshold', type=int, default=2, help='')


parser.add_argument("--reload_path", type=str, default="/mnt/PSO-GAN-4-11/dumped/debug/best-result-non-refine/ru-en/best_mapping2.pth", help="")
# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()

assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
# build model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping_AB, mapping_BA, position_AB, position_BA, discriminator_A, discriminator_B = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping_AB, mapping_BA, position_AB, position_BA, discriminator_A, discriminator_B, params)
evaluator = Evaluator(trainer)



"""
Learning loop for Procrustes Iterative Refinement
"""
if params.n_refinement > 0:
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    for i in range(1):
        trainer.reload_best(params.reload_path)
    #en_ru_only_unsupervised_candinum=1_test1 是保存最强en-ru的未refine
        # training loop
        for n_iter in range(params.n_refinement):

            logger.info('Starting refinement iteration %i...' % n_iter)

            # build a dictionary from aligned embeddings
            trainer.build_dictionary()

            # apply the Procrustes solution
            trainer.procrustes()


            # embeddings evaluation
            to_log = OrderedDict({'n_iter': n_iter})
            evaluator.word_translation(to_log)  # eval source to target
            evaluator.dist_mean_cosine(to_log)

            # JSON log / save best model / end of epoch
            logger.info("__log__:%s" % json.dumps(to_log))
            trainer.save_best(to_log, VALIDATION_METRIC)
            trainer.save_best_refine_acc(to_log,VALIDATION_METRIC_S2T)
            logger.info('End of refinement iteration %i.\n\n' % n_iter)

#classifier = Classifier(params)
#classifier.train()

# export embeddings
# if params.export:
#     trainer.reload_best()
#     trainer.export()


