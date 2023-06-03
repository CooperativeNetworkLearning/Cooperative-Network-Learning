# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch_geometric.nn import GATConv
import os
#from utils import *
from model.layers import *
# from ablation import WOGlobal
# from ablation import WOLocal
# from ablation import WORAGL
# from ablation import baseline
from joblib import Parallel, delayed

class PandemicGNN_GAT(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # arguments setting
        self.adj = data.adj
        self.n0 = data.n0 #节点数
        self.w = args.window
        self.n_layer = args.n_layer

        # self.hidR = args.hidR  # args.hidR好像没用上？？
        #self.hidA = args.hidA
        self.hidP = args.hidP
        self.k = args.k
        self.s = args.s
        self.n = args.n
        self.heads = 2
        #self.res = args.res
        #self.hw = args.hw
        self.droprate = args.dropout
        self.dropout = nn.Dropout(self.droprate)
        self.n0 = data.n0
        self.n0_reality = data.n0_reality
        self.institution_name = args.institution_name
        self.runTag = args.runTag

        # new add
        self.device = args.device
        self.hidR = args.hidR
        self.edge_index = data.edge_index
        self.backbone = RegionAwareConv(P=self.w, m=self.n0, k=self.k, hidP=self.hidP)

        # self.GATBlock = nn.ModuleList(
        #      [GATConv(in_channels=self.hidR, out_channels=self.hidR) for i in range(self.n)])
        self.GATBlock1 = GATConv(in_channels=self.hidR, out_channels=self.hidR, heads=self.heads)
        self.GATBlock2 = GATConv(in_channels=self.hidR*self.heads, out_channels=self.hidR)


        self.output = nn.Linear(self.hidR*(1+1) , self.hidR)
        self.output2 = nn.Linear(self.hidR, 1)


        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)  # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x,  saveEmb=False, shareEmb = None):
        """

        :param x:
        :param trustGNNpath: True表示开启integratedGNN模式
        :param saveEmb: True表示返回嵌入
        :param index: 只选batch中所需的数据的嵌入
        :return:
        """
        # print(index.shape) batch_size
        batch_size = x.shape[0]  # batchsize, w, m
        n0 = self.adj.shape[0]
        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        feat_emb = self.backbone(x)


        if saveEmb:
            # local_feat_emb = feat_emb[:, self.n0_reality, :].unsqueeze(1)
            local_feat_emb = torch.mean(feat_emb[:, :self.n0_reality, :], dim=1).unsqueeze(1)
            return local_feat_emb

        # 将获取的嵌入进行拼接
        if shareEmb is not None:
            feat_emb = torch.cat((feat_emb, shareEmb), dim=1)

        node_state_temp = torch.zeros([feat_emb.shape[0],feat_emb.shape[1],feat_emb.shape[2]*self.heads], device=torch.device(self.device))
        for i in range(batch_size):
            node_state_temp[i, :, :] = self.GATBlock1(feat_emb[i, :, :], self.edge_index)

        node_state = torch.zeros(feat_emb.shape, device=torch.device(self.device))
        for i in range(batch_size):
            node_state[i, :, :] = self.GATBlock2(node_state_temp[i, :, :], self.edge_index)

        # Final prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1) #[128,10,120]


        # decoder
        node_state = F.leaky_relu(node_state)
        node_state = self.dropout(node_state)
        res = self.output(node_state)
        res = F.leaky_relu(res)
        res = self.dropout(res)
        res = self.output2(res)
        #res = F.leaky_relu(res)
        res = res.squeeze(2)

        return res



