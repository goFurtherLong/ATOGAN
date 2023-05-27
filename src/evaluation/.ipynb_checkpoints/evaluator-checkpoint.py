# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor

from .wordsim import get_wordsim_scores, get_crosslingual_wordsim_scores, get_wordanalogy_scores
from .word_translation import get_word_translation_accuracy
from .sent_translation import load_europarl_data, get_sent_translation_accuracy
from ..dico_builder import get_candidates, build_dictionary
from src.utils import get_idf
from .word_translation import get_nn_avg_dist
import torch
logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping_AB = trainer.mapping_AB
        self.mapping_BA = trainer.mapping_BA
        self.position_AB = trainer.position_AB
        self.position_BA = trainer.position_BA
        self.discriminator_A = trainer.discriminator_A
        self.discriminator_B = trainer.discriminator_B
        self.params = trainer.params


 


    def word_translation(self, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        raw_src_emb = self.src_emb.weight.data
        src_emb = self.mapping_AB(self.position_AB(self.src_emb.weight)).data
        tgt_emb = self.tgt_emb.weight.data
        #'nn', 'csls_knn_10'
        for method in ['nn','csls_knn_10']:
            results = get_word_translation_accuracy(
                self.src_dico.lang, self.src_dico.word2id, src_emb,
                self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
                raw_src_emb,
                method=method,
                dico_eval=self.params.dico_eval
            )
            to_log.update([('%s-%s' % (k, method), v) for k, v in results])
            k,v = results[0]
            to_log['best-acc-S2T-%s' % (k)] = v
        return v
    
    def word_translation2(self,to_log):
        logger.info("this is the target to source")
        raw_src_emb = self.src_emb.weight.data
        src_emb = self.mapping_BA(self.position_BA(self.tgt_emb.weight)).data
        tgt_emb = self.src_emb.weight.data

        for method in ['csls_knn_10']:
            results = get_word_translation_accuracy(
                self.tgt_dico.lang, self.tgt_dico.word2id, src_emb,
                self.src_dico.lang, self.src_dico.word2id, tgt_emb,
                raw_src_emb,
                method=method,
                dico_eval=self.params.dico_eval
            )
            k,v = results[0]
            to_log['best-acc-T2S-%s' % (k)] = v
        return v


    def dist_mean_csls(self, to_log, knn=1):
        pass
        # get normalized embeddings
        src_emb = self.mapping_AB(self.position_AB(self.src_emb.weight)).data
        tgt_emb = self.tgt_emb.weight.data
        if self.params.normalize_embeddings != "renorm":
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        average_dist1 = get_nn_avg_dist(tgt_emb, src_emb, 10)
        average_dist2 = get_nn_avg_dist(src_emb, tgt_emb, 10)
        average_dist1 = torch.from_numpy(average_dist1).type_as(src_emb)
        average_dist2 = torch.from_numpy(average_dist2).type_as(tgt_emb)

        max_size = 10000
        source_indices = torch.LongTensor(np.arange(min(max_size, src_emb.size(0))))
        if self.params.cuda:
            average_dist1.cuda()
            average_dist2.cuda()
            source_indices = source_indices.cuda()
        source_tensor = src_emb[source_indices]

        batched_list = []
        batched_list_idx = []
        batch_size = 512
        for i in range(0, source_tensor.size(0), batch_size):
            # queries / scores
            query = source_tensor[i:i + batch_size]
            scores = query.mm(tgt_emb.transpose(0, 1))
            rows = query.size(0)
            scores.mul_(2)
            scores.sub_(average_dist1[i:i + rows][:, None])
            scores.sub_(average_dist2[None, :])
            best_scores, best_idx = scores.topk(knn)
            batched_list.append(best_scores)
            batched_list_idx.append(best_idx)

        metric = torch.cat(batched_list, 0).data.cpu().numpy()
        mean_metric = np.mean(metric)
        logger.info("Mean csls (%s method,  %i max size): %.5f"
                    % ("csls", max_size, mean_metric.item()))
        to_log['mean_csls-%s-%i' % ("csls", max_size)] = mean_metric.item()


    def dist_mean_cosine(self, to_log):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb = self.mapping_AB(self.position_AB(self.src_emb.weight)).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        # build dictionary
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
            logger.info("Mean cosine (%s method, %s build, %i max size): %.5f"
                        % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
            to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine
            to_log['dico_size_s2t'] = dico.size(0)
        return mean_cosine

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        #self.monolingual_wordsim(to_log)
        #self.crosslingual_wordsim(to_log)
        self.word_translation(to_log)
        #self.word_translation2(to_log)
        self.dist_mean_cosine(to_log)
        #self.eval_dis(to_log)

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        src_preds = []
        tgt_preds = []

        self.discriminator_B.eval()

        for i in range(0, self.src_emb.num_embeddings, bs):
            emb = Variable(self.src_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator_B(self.mapping_AB(self.position_AB(emb))).view(-1)
            src_preds.extend(preds.data.cpu().tolist())

        for i in range(0, self.tgt_emb.num_embeddings, bs):
            emb = Variable(self.tgt_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator_B(emb).view(-1)
            tgt_preds.extend(preds.data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: %.5f / %.5f"
                    % (src_pred, tgt_pred))

        src_accu = np.mean([x >= 0.5 for x in src_preds])
        tgt_accu = np.mean([x < 0.5 for x in tgt_preds])
        dis_accu = ((src_accu * self.src_emb.num_embeddings + tgt_accu * self.tgt_emb.num_embeddings) /
                    (self.src_emb.num_embeddings + self.tgt_emb.num_embeddings))
        logger.info("Discriminator source / target / global accuracy: %.5f / %.5f / %.5f"
                    % (src_accu, tgt_accu, dis_accu))

        to_log['dis_accu'] = dis_accu
        to_log['dis_src_pred'] = src_pred
        to_log['dis_tgt_pred'] = tgt_pred
