#encoding:utf-8
# -----------------------------------------------------------
# "Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking"
# WangTan, XingXu, YangYang, Alan Hanjalic, HengtaoShen, JingkuanSong
# ACM Multimedia 2019, Nice, France
# Writen by WangTan, 2019
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from utils import collect_match, collect_neg, acc_train
import seq2vec


class AbstractNoAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[]):
        super(AbstractNoAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words

        self.num_classes = 1
        # Modules
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])

        # Modules for classification
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_h'], self.num_classes)

        self.Eiters = 0

    def _fusion(self, input_v, input_q):
        raise NotImplementedError


    def _classif(self, x):

        batch_size_v = x.size(0)
        batch_size_t = x.size(1)

        if 'activation' in self.opt['classif']:
            x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x,
                      p=self.opt['classif']['dropout'],
                      training=self.training)
        x = self.linear_classif(x)
        x = torch.sigmoid(x)
        x = x.view( batch_size_v, batch_size_t)
        return x


    def forward(self, input_v, input_t):
        if input_v.dim() != 4 and input_t.dim() != 2:
            raise ValueError
        self.Eiters += 1
        batch_size = input_v.size(0)

        x_t_vec = self.seq2vec(input_t)
        x_v = torch.mean(input_v, 1)
        x = self._fusion(x_v, x_t_vec)
        x = self._classif(x)


        # """calculate acc during training"""
        # if self.training:
        #     acc, recall, precision = acc_train(x_m.cpu().clone().data)
        #     self.logger.update('acc', acc)
        #     self.logger.update('rec', recall)
        #     self.logger.update('pre', precision)

        return x




class FusionNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[]):
        opt['fusion']['dim_h'] = opt['fusion']['dim_mm']
        super(FusionNoAtt, self).__init__(opt, vocab_words)
        # Modules for classification
        self.fusion = Core_Fusion(self.opt['fusion'])

    def _fusion(self, x_v, x_t):
        return self.fusion(x_v, x_t)



class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_t):
        raise NotImplementedError('input should be visual and language')



class Core_Fusion(AbstractFusion):

    def __init__(self, opt):
        super(Core_Fusion, self).__init__(opt)

        # visul & text embedding
        self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_hv'])
        self.linear_t = nn.Linear(self.opt['dim_t'], self.opt['dim_ht'])

        # Core tensor
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['dim_hv'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

        self.list_linear_ht = nn.ModuleList([
            nn.Linear(self.opt['dim_ht'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])


    def forward(self, input_v, input_t):
        if input_v.dim() != input_t.dim() and input_v.dim() != 3:
            raise ValueError
        batch_size_v = input_v.size(0)
        batch_size_t = input_t.size(0)

        x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
        x_v = self.linear_v(x_v)
        if 'activation_v' in self.opt:
            x_v = getattr(F, self.opt['activation_v'])(x_v)

        x_t = F.dropout(input_t, p=self.opt['dropout_t'], training=self.training)
        x_t = self.linear_t(x_t)
        if 'activation_t' in self.opt:
            x_t = getattr(F, self.opt['activation_t'])(x_t)


        x_mm = []

        for i in range(self.opt['R']):

            x_hv = F.dropout(x_v, p=self.opt['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            if 'activation_hv' in self.opt:
                x_hv = getattr(F, self.opt['activation_hv'])(x_hv)


            x_ht = F.dropout(x_t, p=self.opt['dropout_ht'], training=self.training)
            x_ht = self.list_linear_ht[i](x_ht)
            if 'activation_ht' in self.opt:
                x_ht = getattr(F, self.opt['activation_ht'])(x_ht)


            x_mm.append(torch.mul(x_hv[:, None, :], x_ht[None, :, :]))


        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size_v, batch_size_t, self.opt['dim_mm'])

        if 'activation_mm' in self.opt:
            x_mm = getattr(F, self.opt['activation_mm'])(x_mm)

        return x_mm




class Fusion2d(Core_Fusion):

    def __init__(self, opt):
        super(Fusion2d, self).__init__(opt)

    def forward(self, input_v, input_t):
        if input_v.dim() != input_t.dim() and input_v.dim() != 3:
            raise ValueError
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        dim_hv = input_v.size(2)
        dim_ht = input_t.size(2)
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_t.is_contiguous():
            input_t = input_t.contiguous()
        x_v = input_v.view(batch_size * weight_height, self.opt['dim_hv'])
        x_t = input_t.view(batch_size * weight_height, self.opt['dim_ht'])
        x_mm = super().forward(x_v, x_t)
        if not x_mm.is_contiguous():
            x_mm = x_mm.contiguous()
        x_mm = x_mm.view(batch_size, batch_size, weight_height, self.opt['dim_mm'])
        return x_mm



def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = FusionNoAtt(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model