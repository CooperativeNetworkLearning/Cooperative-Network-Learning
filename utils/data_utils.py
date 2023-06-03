import sys
import torch
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable
import scipy.sparse as sp
import sklearn
from sklearn import metrics
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
from model.PandemicGNN import *

class TrustGNNLoader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.w = args.window  # 20
        self.h = args.horizon  # 1
        self.d = 0
        self.add_his_day = False

        #self.save_dir = args.save_dir
        self.localIndex = args.localIndex
        self.device = torch.device(args.device)

        self.rawdat = np.loadtxt(open("../data/{}.txt".format(args.dataset)), delimiter=',')
        # 只有机构内部的节点数据是可见的
        if self.localIndex:
            self.rawdat = self.rawdat[:, self.localIndex]
        self.n_sample, self.n0_reality = self.rawdat.shape  # n_sample, n_group
        print('data shape', self.n_sample, self.n0_reality)  # 785, 10


        self.n0 = self.n0_reality
        self.global_nei_num = args.global_nei_num

        if args.sim_mat:
            #self.load_sim_mat(args)
            self.adj = torch.ones((self.n0+self.global_nei_num, self.n0+self.global_nei_num), device = self.device)
            adj_total = torch.Tensor(np.loadtxt(open("../data/{}.txt".format(args.sim_mat)), delimiter=','))
            self.adj[:self.n0_reality, :self.n0_reality] = adj_total[self.localIndex, :][:,self.localIndex]
            self.degree_adj = torch.sum(self.adj, dim=-1)

            sp_adj = sp.coo_matrix(self.adj.cpu())
            self.edge_index = torch.LongTensor(np.vstack((sp_adj.row, sp_adj.col))).to(self.device)


        if (len(self.rawdat.shape) == 1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        self.dat = np.zeros(self.rawdat.shape)  # 785,10


        self.scale = np.ones(self.n0)

        self._pre_train(int(args.train * self.n_sample), int((args.train + args.val) * self.n_sample), self.n_sample)  # 392,549,785
        self._split(int(args.train * self.n_sample), int((args.train + args.val) * self.n_sample), self.n_sample)
        print('size of train/val/test sets', len(self.train[0]), len(self.val[0]), len(self.test[0]))

        self.totalData = self._batchify(range(self.w + self.h - 1, self.n_sample), self.h) #[761,20,11]

    def load_label_file(self, filename):
        labelfile = pd.read_csv("data/" + filename + ".csv", header=None)
        labelLen = len(labelfile)
        label = dict()
        for i in range(labelLen):
            label[labelfile.iloc[i, 0]] = labelfile.iloc[i, 1]
        return label, labelLen

    def load_external(self, args):
        label, label_num = self.load_label_file(args.label)
        files = os.listdir("data/{}".format(args.extra))
        filesLen = len(files)
        extra_adj_list = []
        for i in range(filesLen):
            snapshot = pd.read_csv("data/" + args.extra + "/" + files[i], header=None)
            snapshot_len = len(snapshot)
            extra_adj = np.zeros((label_num, label_num))
            for j in range(snapshot_len):
                extra_adj[label[snapshot.iloc[j, 0]], label[snapshot.iloc[j, 1]]] = snapshot.iloc[j, 2]
            # print(extra_adj)
            extra_adj_list.append(extra_adj)
        extra_adj = torch.Tensor(np.array(extra_adj_list))
        print('external information', extra_adj.shape)
        self.external = Variable(extra_adj)
        if args.cuda:
            self.external = extra_adj.cuda()

    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(open("../data/{}.txt".format(args.sim_mat)), delimiter=','))
        self.degree_adj = torch.sum(self.adj, dim=-1)
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()
            self.degree_adj = self.degree_adj.cuda()

    def _pre_train(self, train, valid, test):
        # self.w = args.window 20
        # self.h  = args.horizon 5
        self.train_set = train_set = range(self.w + self.h - 1, train)
        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n_sample)

        # 数据划分
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)

        # 数据集归一化
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy()  # 199, 47
        self.max = np.max(train_mx, 0)#[:self.n0_reality]
        self.min = np.min(train_mx, 0)#[:self.n0_reality]
        self.peak_thold = np.mean(train_mx, 0)#[:self.n0_reality]
        self.dat = (self.rawdat - self.min) / (self.max - self.min + 1e-12)
        print("_pre_train:", self.dat.shape)

    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h)  # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if train == valid:
            self.val = self.test

    def _batchify(self, idx_set, horizon, useraw=False):  ###tonights work

        n = len(idx_set)
        Y = torch.zeros((n, self.n0))  # 368,10
        X = torch.zeros((n, self.w, self.n0))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.w

            if useraw:  # for narmalization
                X[i, :self.w, :] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i, :] = torch.from_numpy(self.rawdat[idx_set[i], :])
            else:
                his_window = self.dat[start:end, :]
                if self.add_his_day:
                    if idx_set[i] > 51:  # at least 52
                        his_day = self.dat[idx_set[i] - 52:idx_set[i] - 51, :]  #
                    else:  # no history day data
                        his_day = np.zeros((1, self.n0))

                    his_window = np.concatenate([his_day, his_window])
                    # print(his_window.shape,his_day.shape,idx_set[i],idx_set[i]-52,idx_set[i]-51)
                    X[i, :self.w + 1, :] = torch.from_numpy(his_window)  # size (window+1, m)
                else:
                    X[i, :self.w, :] = torch.from_numpy(his_window)  # size (window, m)
                Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])

        return [X, Y]

    # original
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt, :]
            Y = targets[excerpt, :]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()

            data = [X, Y, excerpt]
            yield data
            start_idx += batch_size

def get_net(model):
    if model == 'GAT':
        net = PandemicGNN_GAT
    elif model == 'SAGE':
        net = PandemicGNN_SAGE

    return net





def gen_institution(dataset_name, clientNum=None):
    # args.dataset: adjacency matrix filename (*-adj.txt)
    # 机构划分列表
    if dataset_name == 'state360':
        sim_mat = 'state-adj-49'
        # 这个是通过谱划分得到的
        if clientNum == 3:
            institution_list = [[1, 2, 4, 5, 9, 10, 14, 21, 24, 25, 26, 29, 32, 35, 39, 42, 45, 48],
                                [0, 3, 8, 11, 12, 13, 15, 16, 18, 20, 22, 23, 31, 33, 34, 38, 40, 41, 44, 46, 47],
                                [6, 7, 17, 19, 27, 28, 30, 36, 37, 43]]

        elif clientNum == 4 or None:
            # 这个是根据地理位置划分的，（少了AK，HI两个州）
            # 按照美国地理区域划分为4个机构
            # Northeast, Midwest, South, West
            institution_list_org = [[1,4,8,9,16,17,19,23,32,35,39,41,42,45,47],[3,5,6,11,25,27,30,36,43,46,49],
                                    [7,18,20,28,29,31,37,38,44],[12,13,14,15,21,22,24,26,33,34,40,48]]
            institution_list = []
            for idx in institution_list_org:
                t = np.array(idx) - 1
                institution_list.append(t.tolist())

    elif dataset_name == 'region785':
        sim_mat = 'region-adj'
        institution_list = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9]]
    elif dataset_name == 'japan':
        sim_mat = 'japan-adj'
        institution_list = [[0, 1, 2, 4, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37,
             39, 40, 41, 42, 43, 46], [3, 5, 10, 19, 23, 45], [7, 11, 24, 25, 31, 38, 44]]
    elif dataset_name == 'spain':
        sim_mat = 'spain-adj'
        institution_list = [[1, 2, 3, 4, 7, 9, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 31, 32, 34, 35, 43, 44, 45, 46,
             47, 48, 51], [0, 5, 6, 8, 11, 14, 27, 28, 30, 36, 37, 39, 40, 42, 49, 50], [10, 16, 33, 38, 41]]
    elif dataset_name == 'australia':
        sim_mat = 'australia-adj'
        institution_list = [[0, 1, 6], [2, 4, 7], [3, 5]]
    elif dataset_name == 'china':
        sim_mat = 'china-adj'
        institution_list = [[0, 1, 2, 3, 4, 5, 6, 7, 15, 29], [19, 21, 22, 23, 24, 25, 26, 27, 28, 30],
                            [8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20]]
    elif dataset_name == 'monkeypox':
        sim_mat = 'monkeypox-adj'
        institution_list = [[7, 13, 16, 17, 19, 22], [0, 2, 3, 4, 5, 11, 12, 15, 24, 25, 26], [1, 6, 8, 9, 10, 14, 18, 20, 21, 23, 27, 28]]
    elif dataset_name == 'twitter':
        sim_mat = 'twitter-adj'
        # 通过谱聚类划分为3个机构
        if clientNum == 3:
            institution_list = [[0, 2, 7, 8, 10, 11, 14, 15, 19, 21, 22, 30, 32, 37, 39, 40, 43, 45, 46],
                              [1, 3, 4, 9, 12, 13, 20, 23, 24, 25, 28, 31, 33, 34, 38, 41, 44, 47],
                              [5, 6, 16, 17, 18, 26, 27, 29, 35, 36, 42]] #更新后的
        elif clientNum == 4 or None:
            # 按照美国地理区域划分为4个机构
            # Northeast, Midwest, South, West
            institution_list_org = [[6, 17, 19, 27, 28, 30, 36, 37, 43], [11, 12, 13, 14, 20, 21, 23, 25, 32, 33, 39, 47],
                                   [1, 3, 7, 8, 9, 15, 16, 18, 22, 31, 34, 38, 40, 41, 44, 46],
                                   [2, 4, 5, 10, 24, 26, 29, 35, 42, 45, 48]]
            institution_list = []
            for idx in institution_list_org:
                t = np.array(idx) - 1
                institution_list.append(t.tolist())
    elif dataset_name == 'ba1000':
        sim_mat = 'ba1000_adj'
        institution_list = [[2, 9], [1, 3, 5], [7, 0], [4, 6, 8]]
    elif dataset_name == 'er1000':
        sim_mat = 'er1000_adj'
        institution_list = [[1, 6, 7], [0, 2, 4], [5, 9], [3, 8]]
    elif dataset_name == 'er10000':
        sim_mat = 'er10000_adj'
        institution_list = [[1, 2], [0, 6], [4, 7, 9], [3, 5, 8]]
    elif dataset_name == 'ba10000':
        sim_mat = 'ba10000_adj'
        institution_list = [[0, 3, 6], [1, 7, 8], [2, 4], [5, 9]]
    elif dataset_name == 'er100003' or 'er100003_threshold' or 'er100003_cml' or 'er100003_SIS':
        sim_mat = 'er100003_adj'
        institution_list = [[3, 5, 12, 13, 15, 18, 20, 22, 25, 28], [9, 16, 21, 27, 29], [1, 7, 11, 19],
                            [2, 4, 6, 14, 23, 26],
                            [0, 8, 10, 17, 24]]
    elif dataset_name == 'ba100003' or 'ba100003_threshold' or 'ba100003_cml' or 'ba100003_SIS':
        sim_mat = 'ba100003_adj'
        institution_list = [[20, 24, 25], [5, 8, 9, 17, 21], [10, 18, 23], [1, 11, 13, 14, 22, 28],
                            [0, 2, 3, 4, 6, 7, 12, 15, 16, 19, 26, 27, 29]]

    return sim_mat, institution_list


def regression_metrics(y_true, y_pred):

    metrics={}
    metrics['mse'] = sklearn.metrics.mean_squared_error(y_true, y_pred).tolist()
    metrics['rmse'] = np.sqrt(metrics['mse']).tolist()
    metrics['mae'] = sklearn.metrics.mean_absolute_error(y_true, y_pred).tolist()
    metrics['mape'] = np.mean(np.abs((y_pred - y_true) / (y_true + 0.00001))).tolist()
    metrics['pcc'] = pearsonr(y_true, y_pred)[0].tolist()
    metrics['r2'] = sklearn.metrics.r2_score(y_true, y_pred, multioutput='uniform_average').tolist()

    return metrics

