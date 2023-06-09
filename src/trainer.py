# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import math
from logging import getLogger
import scipy
import scipy.linalg
from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .dico_builder import build_dictionary,get_candidates
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
import itertools
from torch.autograd import Variable
import torch
import copy
from copy import deepcopy

# import pandas as pd

logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping_AB, mapping_BA, position_AB, position_BA, discriminator_A, discriminator_B, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.align_dico = params.align_dico # align_dico is Tensor of size (n,2),    eg: ( idx1 , idx2 )
        self.valid_dico = params.valid_dico # same to align_dico, but with validation  data.
        self.params = params
        

        #network
        self.mapping_AB = mapping_AB
        self.mapping_BA = mapping_BA
        self.position_AB = position_AB
        self.position_BA = position_BA
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B
        
        #self.optimizer_G = torch.optim.SGD(itertools.chain(mapping_AB.parameters(), mapping_BA.parameters(), position_AB.parameters(),position_BA.parameters()),lr=params.lr)
        self.optimizer_G = torch.optim.SGD(itertools.chain(mapping_AB.parameters(), mapping_BA.parameters()),lr=params.lr)
        self.optimizer_D1 = torch.optim.SGD(itertools.chain(discriminator_B.parameters(), discriminator_A.parameters()),lr=params.dis_lr)
 


        G_candidate = []
        mapping_AB  = copy.deepcopy(self.mapping_AB.state_dict())
        mapping_BA  = copy.deepcopy(self.mapping_BA.state_dict())
        discriminator_A  = copy.deepcopy(self.discriminator_A.state_dict())
        discriminator_B  = copy.deepcopy(self.discriminator_B.state_dict())
        optimizer_G   = copy.deepcopy(self.optimizer_G.state_dict())
        G_candidate.append(mapping_AB)
        G_candidate.append(mapping_BA)
        G_candidate.append(discriminator_A)
        G_candidate.append(discriminator_B)
        self.G_candis = G_candidate
        self.optG_candis = optimizer_G


        # Lossess
        self.criterion_GAN = torch.nn.BCELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.criterion_l1loss = torch.nn.L1Loss()
        self.criterion_LS = torch.nn.MSELoss()


        # # best validation score
        self.best_valid_metric = -1e12
        self.best_valid_S2T_metric = -1e12
        self.best_valid_T2S_metric = -1e12
        #
        self.decrease_lr = False

    def get_dis_xy(self, bs):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        #bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)

        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()
        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, requires_grad=False))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, requires_grad=False))
        src_emb = Variable(src_emb.data, requires_grad=False)
        tgt_emb = Variable(tgt_emb.data, requires_grad=False)

        return src_emb, tgt_emb

    




    def dis_step1(self, stats):
            """
            Train the discriminator.
            """
            self.discriminator_A.train()
            self.discriminator_B.train()
            Tensor = torch.cuda.FloatTensor if self.params.cuda else torch.Tensor
            target_real = Variable(Tensor(self.params.batch_size).fill_(0.9), requires_grad=False)
            target_fake = Variable(Tensor(self.params.batch_size).fill_(0.1), requires_grad=False)
            real_A, real_B = self.get_dis_xy(self.params.batch_size)
            fake_B = self.mapping_AB(self.position_AB(real_A))
            fake_A = self.mapping_BA(self.position_BA(real_B))

            # Real loss
            pred_real_A = self.discriminator_A(real_A)
            loss_D_real_A = self.criterion_GAN(pred_real_A, target_real)

            # Fake loss
            #fake_A = self.mapping_BA(self.position_BA(real_B))
            pred_fake_A = self.discriminator_A(fake_A)
            loss_D_fake_A = self.criterion_GAN(pred_fake_A, target_fake)

            # Total loss
            loss_D_A = loss_D_real_A + loss_D_fake_A

            # Real loss
            pred_real_B = self.discriminator_B(real_B)
            loss_D_real_B = self.criterion_GAN(pred_real_B, target_real)

            # Fake loss
            #fake_B = self.mapping_AB(self.position_AB(real_A))
            pred_fake_B = self.discriminator_B(fake_B)
            loss_D_fake_B = self.criterion_GAN(pred_fake_B, target_fake)

            # Total loss
            loss_D_B = loss_D_real_B + loss_D_fake_B

            loss_D = (loss_D_A + loss_D_B)*0.5
            stats['DIS_COSTS'].append(loss_D.data.item())
            self.optimizer_D1.zero_grad()
            loss_D.backward()
            self.optimizer_D1.step()          
            ###################################

    def score(self):
        """
        Given source and target word embeddings, and a dictionary,
        evaluate the translation accuracy using the precision@k.
        """
        valid_A, valid_B = self.get_dis_xy(self.params.score_size)
        fake_B = self.mapping_AB(self.position_AB(valid_A))
        fake_A = self.mapping_BA(self.position_BA(valid_B))
        pred_fake_B = self.discriminator_B(fake_B)
        pred_real_B = self.discriminator_B(valid_B)
        pred_fake_A = self.discriminator_A(fake_A)
        pred_real_A = self.discriminator_A(valid_A)
        Fq = torch.sum(pred_fake_B)/pred_fake_B.shape[0] + torch.sum(pred_fake_A/pred_fake_A.shape[0])



        #Fd
        Tensor = torch.cuda.FloatTensor if self.params.cuda else torch.Tensor
        target_real = Variable(Tensor(self.params.score_size).fill_(0.9), requires_grad=False)
        target_fake = Variable(Tensor(self.params.score_size).fill_(0.1), requires_grad=False)

        eval_D_fake = self.criterion_GAN(pred_fake_B,target_fake) 
        eval_D_real = self.criterion_GAN(pred_real_B,target_real)
        eval_D = eval_D_fake + eval_D_real
        gradients = torch.autograd.grad(outputs=eval_D, inputs=self.discriminator_B.parameters(),
                                        grad_outputs=torch.ones(eval_D.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)#eval_D.size().cuda()
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad,grad]) 
        Fd1 = -torch.log(torch.norm(allgrad)).data.cpu().numpy()

        eval_D_fake = self.criterion_GAN(pred_fake_A, target_fake)
        eval_D_real = self.criterion_GAN(pred_real_A, target_real)
        eval_D = eval_D_fake + eval_D_real
        gradients = torch.autograd.grad(outputs=eval_D, inputs=self.discriminator_A.parameters(),
                                        grad_outputs=torch.ones(eval_D.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)  # eval_D.size().cuda()
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad, grad])
        Fd2 = -torch.log(torch.norm(allgrad)).data.cpu().numpy()
        Fd = Fd1 + Fd2
        return ((Fq + self.params.lambda_f* Fd)*0.5).data.cpu().numpy()

    
    def recover_old_parameters(self):
        self.mapping_AB.load_state_dict(self.G_candis[0])
        self.mapping_BA.load_state_dict(self.G_candis[1])
        self.discriminator_A.load_state_dict(self.G_candis[2])
        self.discriminator_B.load_state_dict(self.G_candis[3])
        self.optimizer_G.load_state_dict(self.optG_candis)

    def mapping_step(self, stats, lambda_w):
        """
        Fooling discriminator training step.
        """
        self.discriminator_A.eval()
        self.discriminator_B.eval()

        real_A, real_B = self.get_dis_xy(self.params.batch_size)

        Tensor = torch.cuda.FloatTensor if self.params.cuda else torch.Tensor
        ###### Generators A2B and B2A ######
        target_real = Variable(Tensor(self.params.batch_size).fill_(0.9), requires_grad=False)
        target_fake = Variable(Tensor(self.params.batch_size).fill_(0.1), requires_grad=False)

        # GAN loss  --1
        fake_B = self.mapping_AB(self.position_AB(real_A))
        pred_fake_B = self.discriminator_B(fake_B)
        # pred_real = self.discriminator_B(real_B)
        loss_GAN_A2B1 = self.criterion_GAN(pred_fake_B, target_real)

        fake_A = self.mapping_BA(self.position_BA(real_B))
        pred_fake_A = self.discriminator_A(fake_A)
        # pred_real = self.discriminator_A(real_A)
        loss_GAN_B2A1 = self.criterion_GAN(pred_fake_A, target_real)
        loss_1 = (loss_GAN_A2B1 + loss_GAN_B2A1) * 0.5

        # Cycle loss --2
        recovered_A = self.mapping_BA(self.position_BA(fake_B))
        loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)

        recovered_B = self.mapping_AB(self.position_AB(fake_A))
        loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)
        loss_2 = (loss_cycle_ABA + loss_cycle_BAB) * 0.5

        # Identity loss --3
        # G_A2B(B) should equal B if real B is fed
        same_B = self.mapping_AB(self.position_AB(real_B))
        loss_identity_B = self.criterion_identity(same_B, real_B)
        # G_B2A(A) should equal A if real A is fed
        same_A = self.mapping_BA(self.position_BA(real_A))
        loss_identity_A = self.criterion_identity(same_A, real_A)
        loss_3 = (loss_identity_B + loss_identity_A) * 0.5

        #rsgan loss --4
        pred_real_A = Variable(self.discriminator_A(real_A), requires_grad=False)
        pred_real_B = Variable(self.discriminator_B(real_B), requires_grad=False)
        loss_GAN_A2B2 = self.criterion_GAN(pred_fake_B, pred_real_B)
        loss_GAN_B2A2 = self.criterion_GAN(pred_fake_A, pred_real_A)
        loss_4 = (loss_GAN_A2B2 + loss_GAN_B2A2) * 0.5

        #lsgan loss --5
        loss_GAN_A2B3 = self.criterion_LS(pred_fake_B, target_real)
        loss_GAN_B2A3 = self.criterion_LS(pred_fake_A, target_real)
        loss_5 = (loss_GAN_A2B3 + loss_GAN_B2A3) * 0.5

        #minimax loss --6
        #total loss
        loss_G = lambda_w[0]*loss_1  +lambda_w[3]*loss_4 +lambda_w[4]*loss_5 + lambda_w[1]*loss_2 + lambda_w[2]*loss_3
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        ###################################
        self.orthogonalize()


        stats['GEN_COSTS'].append(loss_G.data.item())
        #stats['sup/unsup'].append(loss_rate)

        return 2 * self.params.batch_size

    # select topK child as the evolution result, and store the result's parameters in "G_list", "optG_list" 
    def  update_G_candi(self):
        G_candidate = []
        mapping_AB  = copy.deepcopy(self.mapping_AB.state_dict())
        mapping_BA  = copy.deepcopy(self.mapping_BA.state_dict())
        discriminator_A  = copy.deepcopy(self.discriminator_A.state_dict())
        discriminator_B  = copy.deepcopy(self.discriminator_B.state_dict())
        optimizer_G   = copy.deepcopy(self.optimizer_G.state_dict())
        G_candidate.append(mapping_AB)
        G_candidate.append(mapping_BA)
        G_candidate.append(discriminator_A)
        G_candidate.append(discriminator_B)
        self.G_candis = G_candidate
        self.optG_candis = optimizer_G




    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping_AB(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)
    

        

    
    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping_AB.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


  


    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping_AB.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
            W2 = self.mapping_BA.weight.data
            beta = self.params.map_beta
            W2.copy_((1 + beta) * W2 - beta * W2.mm(W2.transpose(0, 1).mm(W2)))


    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return

        old_lr_G = self.optimizer_G.param_groups[0]['lr']
        new_lr_G = max(self.params.min_lr, old_lr_G * self.params.lr_decay)
        if new_lr_G < old_lr_G:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr_G, new_lr_G))
            for i in range(self.params.candi_num):
                self.optG_candis["param_groups"][0]['lr'] = new_lr_G
            

    
        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:

            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    for i in range(self.params.candi_num):
                        old_lr = self.optG_candis["param_groups"][0]['lr']
                        self.optG_candis["param_groups"][0]['lr'] *= self.params.lr_shrink
                        logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.optG_candis["param_groups"][0]['lr']))
                    self.decrease_lr = False
                else:
                    self.decrease_lr = True
            else:
                self.decrease_lr = False



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
            G = self.mapping_AB.weight.data.cpu().numpy()
            F = self.mapping_BA.weight.data.cpu().numpy()
            posAB = self.position_AB.weight.data.cpu().numpy()
            posBA = self.position_BA.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(G, path)
            path = os.path.join(self.params.exp_path, 'best_mapping2.pth')
            torch.save(F, path)
            logger.info('* Saving the mapping2 to %s ...' % path)
            path = os.path.join(self.params.exp_path, 'best_posAB.pth')
            torch.save(posAB,path)
            logger.info('* Saving the posAB to %s ...' % path)
            path = os.path.join(self.params.exp_path, 'best_posBA.pth')
            logger.info('* Saving the posBA to %s ...' % path)
            torch.save(posBA,path)
            return True
        return False


            
    def save_best_acc(self, to_log, metric_S2T,metric_T2S):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric_S2T] > self.best_valid_S2T_metric:
            # new best mapping
            self.best_valid_S2T_metric = to_log[metric_S2T]
            logger.info('* Best value for "%s": %.5f' % (metric_S2T, to_log[metric_S2T]))
            # save the mapping
            G = self.mapping_AB.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_acc_mapping_S2T.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(G, path)

        if to_log[metric_T2S] > self.best_valid_T2S_metric:
            # new best mapping
            self.best_valid_T2S_metric = to_log[metric_T2S]
            logger.info('* Best value for "%s": %.5f' % (metric_T2S, to_log[metric_T2S]))
            # save the mapping
            F = self.mapping_BA.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_acc_mapping_T2S.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            
            torch.save(F, path)
            
    def save_best_refine_acc(self, to_log, metric_S2T):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric_S2T] > self.best_valid_S2T_metric:
            # new best mapping
            self.best_valid_S2T_metric = to_log[metric_S2T]
            logger.info('* Best value for "%s": %.5f' % (metric_S2T, to_log[metric_S2T]))
            # save the mapping
            G = self.mapping_AB.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_refine_acc_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(G, path)
            


    def reload_best_refine(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        print('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping_AB.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

        path = os.path.join(self.params.exp_path, 'best_mapping2.pth')
        print('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping_BA.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

       
    def reload_best_acc_refine(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_acc_mapping_S2T.pth')
        print('* Reloading the best_acc model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping_AB.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))


        
    def reload_best(self, exp_path):
        """
        Reload the best mapping.
        """
        #path = os.path.join(exp_path, 'best_acc_mapping_S2T.pth')
        print('* Reloading the best_acc model from %s ...' % exp_path)
        # reload the model
        assert os.path.isfile(exp_path)
        to_reload = torch.from_numpy(torch.load(exp_path))
        W = self.mapping_AB.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))


    def export(self):
        """
        Export embeddings.
        """
        params = self.params
        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        print("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            x = Variable(src_emb[k:k + bs], volatile=True)
            src_emb[k:k + bs] = self.mapping_AB(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)
