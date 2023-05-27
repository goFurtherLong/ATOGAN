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
from typing import List, Mapping
import numpy as np
import torch
from torch import nn
from copy import deepcopy
#from src.classifier import Classifier
from src.utils import bool_flag, initialize_exp, normalize_embeddings
from src.models import build_model
from src.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.dico_builder import get_candidates, build_dictionary
from sklearn.cluster import KMeans
import scipy
from torch.autograd import Variable
from src.evaluation.word_translation import get_word_translation_accuracy, get_word_pairs
from src.utils import get_nn_avg_dist
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'
VALIDATION_METRIC_S2T = 'best-acc-S2T-precision_at_1'
def get_closest_csls(r_src_emb, align_src_emb,k=15):
    emb1 = r_src_emb
    emb2 = align_src_emb
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    knn = 10
    average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
    average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
    average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
    # queries / scores
    query = emb1
    scores = query.mm(emb2.transpose(0, 1))
    scores.mul_(2)
    scores.sub_(average_dist1[:, None])
    scores.sub_(average_dist2[None, :])
    top_matched_content, top_matched_idx = scores.topk(k, 1, True)

    return top_matched_content, top_matched_idx

def get_closest_csls_v2(r_src_emb, align_src_emb,retrieval_k_num=15):
    emb1 = r_src_emb
    emb2 = align_src_emb
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    knn=10  #10
    average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
    average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
    average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)

    # queries / scores
    bs = 4096
    total = emb1.shape[0]
    retrieval_result_idx  =torch.empty([0,retrieval_k_num],dtype=torch.long)
    retrieval_result_content  =torch.empty([0,retrieval_k_num])
    retrieval_result_idx = retrieval_result_idx.cuda()
    retrieval_result_content = retrieval_result_content.cuda()
    for i in range(0, total, bs):
        query = emb1[i:i+bs]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[i:i+bs][:, None])
        scores.sub_(average_dist2[None, :])
        top_matched_content, top_matched_idx = scores.topk(retrieval_k_num, 1, True)
        
        retrieval_result_content = torch.cat((retrieval_result_content,top_matched_content),0)
        retrieval_result_idx = torch.cat((retrieval_result_idx,top_matched_idx),0)


    return retrieval_result_content, retrieval_result_idx

def get_closest_cosine(r_src_emb, align_src_emb, k):
    emb1 = r_src_emb
    emb2 = align_src_emb
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    scores = emb1.mm(emb2.transpose(0, 1))
    #scores = (torch.exp( 10 * scores) - 1)
    top_matched_content, top_matched_idx = scores.topk(k, 1, True)
    return top_matched_content, top_matched_idx

def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])

def instance_map_torch_v2(r_src_emb, align_src_emb,align_tgt_emb, k_closest = 70, m = 10, step_size=0.1):
    print("dtype: ",r_src_emb.dtype)
    diff = (-align_src_emb +align_tgt_emb)
    logger.info( "computing bias")
    move_vector  = torch.empty([0,r_src_emb.shape[1]])
    move_vector =  move_vector.cuda()
    top_matched_weight, top_matched_idx = get_closest_csls_v2(r_src_emb, align_src_emb,retrieval_k_num= k_closest)
    zero_matrix = torch.zeros(top_matched_weight.shape)
    zero_matrix = zero_matrix.cuda()
    top_matched_weight = torch.where(top_matched_weight<0.4,zero_matrix, top_matched_weight)
    bs = 4096
    total = r_src_emb.shape[0]
    for i in range(0, total, bs):
        top_matched_idx_part = top_matched_idx[i:i+bs]
        top_matched_weight_part= top_matched_weight[i:i+bs]
        #logger.info(top_matched_weight_part)
        #bias shape:(bs,k_closest,300)
        bias = diff[top_matched_idx_part.type(torch.int64)]
        #csls_weight_part shape:(bs,300)
        csls_weight_part = torch.exp(m*top_matched_weight_part)
        csls_weight_part = csls_weight_part.div_(csls_weight_part.norm(1, 1, keepdim=True).expand_as(csls_weight_part))
        #move_vector_part shape:(bs,300)
        move_vector_part = torch.sum(csls_weight_part.unsqueeze(2) * bias,dim=1)
        #print("move_vector_part_shape: ",move_vector_part.shape)
        #print("weight_shape: ",move_vector.shape)
        move_vector = torch.cat((move_vector,move_vector_part),0)

    logger.info("updating the emb")
    result = []
    #for j in step_size:
        #print(j)
    new_embs_src = r_src_emb + step_size * move_vector
    result.append(new_embs_src.cpu() if params.cuda else new_embs_src )
    return result

def instance_map(embs_src, align_src_emb,align_tgt_emb, k_closest = 15, step_size = [0.1]):
    
    embs_src = embs_src.cpu().numpy()
    align_src_emb = align_src_emb.cpu().numpy()
    align_tgt_emb =align_tgt_emb.cpu().numpy()
    
    diff_mat = (-align_src_emb +align_tgt_emb)

    ## Mapping/tuning of source space vectors

    logger.info( " Computing dictionary closest neighbours for all source space vectors...")
    embs_src_norm =mat_normalize(embs_src)  
    src_mat_norm = mat_normalize(align_src_emb)

    closest = np.empty(shape=(0,k_closest),dtype=np.intc)
    cosines = np.empty(shape=(0,src_mat_norm.shape[0]),dtype=np.float32)
    print("init shape:",closest.shape)
    bs = 4096*4
    total = embs_src_norm.shape[0]
    for i in range(0, total, bs):
        cosines_part =np.exp( 10 *np.matmul(embs_src_norm[i:i+bs,:], np.transpose(src_mat_norm)) )  -1 
        closest_part = np.flip(np.argsort(cosines_part, axis = 1), axis = 1)[:, :k_closest]
        closest =  np.concatenate((closest, closest_part))
        cosines = np.concatenate((cosines, cosines_part))
        print(closest.shape)

    logger.info(" Tuning/mapping source space vectors...")
    result = []
    for j in step_size:
        proj_src = []
        for i in range(embs_src.shape[0]):
            indices = closest[i]
            cosines_closest = cosines[i][closest[i]]
            cosines_closest = cosines_closest / np.sum(cosines_closest)

            dir_mat = diff_mat[indices, :]
            cos_weights = np.tile(np.reshape(cosines_closest, (len(cosines_closest), 1)), (1, dir_mat.shape[1]))
            
            dir_vec = np.sum(np.multiply(dir_mat, cos_weights), axis = 0)     
            y_pr = embs_src[i] + j * dir_vec
            
            proj_src.append(y_pr)
        result.append(np.array(proj_src))
    return result

# def instance_map_torch(r_src_emb, align_src_emb,align_tgt_emb, k_closest = 15, step_size=0.1):
#     print("dtype: ",r_src_emb.dtype)
#     bias = (-align_src_emb +align_tgt_emb)
#     logger.info( "computing bias")
#     #top_matched_weight, top_matched_idx = get_closest_csls(r_src_emb, align_src_emb, k = k_closest)
#     top_matched_weight, top_matched_idx = get_closest_cosine(r_src_emb, align_src_emb, k = k_closest)
#     print("dtyoe: ",top_matched_weight.dtype)
#     csls_weight = torch.exp(10*top_matched_weight)
#     csls_weight = top_matched_weight
#     csls_weight = csls_weight /torch.sum(csls_weight,dim=1,keepdim=True).expand_as(csls_weight)

#     indices = top_matched_idx.type(torch.int64)
#     bias = bias[indices]
#     print("======================dtype: ",csls_weight.dtype)
#     print("bias shape: ",bias.shape) 
#     logger.info("updating the emb")
#     result = []
    
#     t =  torch.sum(csls_weight.unsqueeze(2) * bias, dim=1) 
#     new_embs_src = r_src_emb + step_size * t
#     result.append(new_embs_src)
#     return result


def instance_map_torch(params,r_src_emb, align_src_emb,align_tgt_emb, k_closest = 70, m = 10, step_size=0.1):
    print("dtype: ",r_src_emb.dtype)
    diff = (-align_src_emb +align_tgt_emb)
    logger.info( "computing bias")
    bs = 4096
    total = r_src_emb.shape[0]
    move_vector  = torch.empty([0,r_src_emb.shape[1]])
    if params.cuda:
        move_vector =  move_vector.cuda()
    for i in range(0, total, bs):
        top_matched_weight_part, top_matched_idx_part = get_closest_cosine(r_src_emb[i:i+bs], align_src_emb, k = k_closest)
        
        if params.control1 == 1:
            #logger.info("use threshold")
            a = torch.zeros(top_matched_weight_part.shape)
            a = a.cuda()
            top_matched_weight_part = torch.where(top_matched_weight_part<params.local_mapping_threshold,a,top_matched_weight_part)
        
        #logger.info(top_matched_weight_part)
        #bias shape:(bs,k_closest,300)
        bias = diff[top_matched_idx_part.type(torch.int64)]
        #csls_weight_part shape:(bs,300)
        csls_weight_part = torch.exp(m*top_matched_weight_part) if params.control2==1 else top_matched_weight_part+0.00001
        
        csls_weight_part = csls_weight_part.div_(csls_weight_part.norm(1, 1, keepdim=True).expand_as(csls_weight_part))
        #move_vector_part shape:(bs,300)
        move_vector_part = torch.sum(csls_weight_part.unsqueeze(2) * bias,dim=1)
        #print("move_vector_part_shape: ",move_vector_part.shape)
        #print("weight_shape: ",move_vector.shape)
        move_vector = torch.cat((move_vector,move_vector_part),0)

    
    logger.info("updating the emb")
    result = []
    #for j in step_size:
        #print(j)
    new_embs_src = r_src_emb + step_size * move_vector
    result.append(new_embs_src.cpu() if params.cuda else new_embs_src )
    return result


def get_dictionary(r_src_emb):
    """
    get the nearst neighbor as  align pairs
    """
    src_emb = r_src_emb.data
    tgt_emb = trainer.tgt_emb.weight.data
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

    # build dictionary
    
    dico = build_dictionary(src_emb, tgt_emb, params)
    return dico

def dist_mean_cosine(params,r_src_emb,tgt_emb ,to_log):
    """
    Mean-cosine model selection criterion.
    """
    # get normalized embeddings
    src_emb = r_src_emb
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

    # build dictionary
    for dico_method in ['csls_knn_10']:
        dico_build = 'S2T'   #hbc mark:maybe try S2T&T2S or S2T|T2S?
        dico_max_size = 10000
        # temp params / dictionary generation
        _params = deepcopy(params)
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
        mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch.Tensor) else mean_cosine
        logger.info("Mean cosine (%s method, %s build, %i max size): %.5f"
                    % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
        to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine

# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data

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

parser.add_argument("--dis_steps", type=int, default=3, help="Discriminator steps")
parser.add_argument("--dis_steps_2", type=int, default=3, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")

# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=150000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--eval_size", type=int, default=32, help="the initial # for each eval, may be changed in trainer.py, will be used in training D")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")

parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.7, help="Shrink the learning rate if the validation metric decreases (1 to disable)")

parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--dis_lr', type=float, default=0.05, help='initial learning rate')
parser.add_argument("--d_cycle_weight", type=float, default=1)
parser.add_argument("--d_weight", type=float, default=0.5)
parser.add_argument("--mono_weight", type=float, default=0.4)
parser.add_argument("--cycle_weight", type=float, default=5.0)

#EVOLUTIONARY GAN 
parser.add_argument("--candi_num", type=int,default=1,help="the num of survived children")
parser.add_argument("--lambda_f", type=float, default=0.3, help = "factor of Fd")

# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=75000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='ru', help="Target language")
parser.add_argument("--src_emb", type=str, default="/mnt/FMGAN/data/wiki.en.vec", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="/mnt/FMGAN/data/wiki.ru.vec", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="renorm,center,renorm", help="Normalize embeddings before training")
parser.add_argument("--reload_path", type=str, default="/mnt/PSO-GAN-4-11/dumped/best-result-refine/en-ru/best_refine_acc_mapping.pth", help="Reload target embeddings")
parser.add_argument("--step_size", type=float, default=0.1)
parser.add_argument("--iter", type=int, default=10)
parser.add_argument("--m", type=int, default=10)
parser.add_argument("--k_closest", type=int, default=10)
parser.add_argument("--control1", type=int, default=1)
parser.add_argument("--control2", type=int, default=1)
parser.add_argument("--local_mapping_threshold", type=float, default=0.4)
# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()

assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
print(params.src_emb)
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
# build model / trainer / evaluator
logger = initialize_exp(params)

path = params.reload_path
print(path)
assert os.path.isfile(path)

src_emb, tgt_emb, mapping_AB, mapping_BA, position_AB, position_BA, discriminator_A, discriminator_B = build_model(params,True)
trainer = Trainer(src_emb, tgt_emb, mapping_AB, mapping_BA, position_AB, position_BA, discriminator_A, discriminator_B, params)
evaluator = Evaluator(trainer)



to_reload = torch.from_numpy(torch.load(path))
W = trainer.mapping_AB.weight.data
assert to_reload.size() == W.size()
W.copy_(to_reload.type_as(W))


to_log = OrderedDict({'n_epoch': -1})


r_src_emb = trainer.mapping_AB(trainer.position_AB(src_emb.weight))


generate_dico = get_dictionary(r_src_emb)
generate_dico = generate_dico.cpu()
align_src_ids = torch.LongTensor(generate_dico[:,0])
align_tgt_ids = torch.LongTensor(generate_dico[:,1])

# get word embeddings
src_emb = r_src_emb.data[align_src_ids]
tgt_emb = trainer.tgt_emb.weight.data[align_tgt_ids]
align_src_emb = src_emb.data
align_tgt_emb = tgt_emb.data

r_src_emb = r_src_emb.data



result = 0
for iter in range(params.iter):
    if iter !=0:
        r_src_emb = result
        generate_dico = get_dictionary(r_src_emb)
        generate_dico = generate_dico.cpu()
        align_src_ids = torch.LongTensor(generate_dico[:,0])
        align_tgt_ids = torch.LongTensor(generate_dico[:,1])

        # get word embeddings
        src_emb = r_src_emb.data[align_src_ids]
        tgt_emb = trainer.tgt_emb.weight.data[align_tgt_ids]
        align_src_emb = src_emb.data
        align_tgt_emb = tgt_emb.data
        r_src_emb = r_src_emb.data
    outputs  =instance_map_torch(params,r_src_emb,align_src_emb,align_tgt_emb, k_closest = params.k_closest, m = params.m, step_size=params.step_size)
    
    #result = torch.from_numpy(outputs[0]) 
    result = outputs[0].float()
    result = result.cuda()
    #eval
    for method in ['nn', 'csls_knn_10']:
        results = get_word_translation_accuracy(
            trainer.src_dico.lang, trainer.src_dico.word2id, result.data,
            trainer.tgt_dico.lang, trainer.tgt_dico.word2id, trainer.tgt_emb.weight.data,
            raw_src_emb="",
            method=method,
            dico_eval=params.dico_eval
        )


#         for (k,v) in results:
#             print(k,v)
#         k,v = results[0]
#         if method in ['csls_knn_10']:
#             to_log['best-acc-S2T-%s' % (k)] = v
#             if to_log['best-acc-S2T-precision_at_1'] > trainer.best_valid_S2T_metric: 
#                 trainer.best_valid_S2T_metric = to_log['best-acc-S2T-precision_at_1']
#                 get_word_pairs(params,trainer.src_dico.lang, trainer.src_dico.word2id, result.data,trainer.tgt_dico.lang, 
#                                               trainer.tgt_dico.word2id, trainer.tgt_emb.weight.data,raw_src_emb="",
#                                               method=method,dico_eval=params.dico_eval)
#     dist_mean_cosine(params,result,tgt_emb ,to_log)
    logger.info('End of epoch %i.\n\n' % iter)

print("==========")


src_path = os.path.join(params.exp_path, 'lm-vectors-%s.pth' % params.src_lang)
tgt_path = os.path.join(params.exp_path, 'lm-vectors-%s.pth' % params.tgt_lang)
logger.info('Writing source embeddings to %s ...' % src_path)
torch.save({'dico': params.src_dico, 'vectors': result.data.cpu()}, src_path)
logger.info('Writing target embeddings to %s ...' % tgt_path)
torch.save({'dico': params.tgt_dico, 'vectors': trainer.tgt_emb.weight.data.cpu()}, tgt_path)





