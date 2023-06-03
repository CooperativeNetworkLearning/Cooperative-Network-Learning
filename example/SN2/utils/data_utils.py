import sys
import torch
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable
import scipy.sparse as sp


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

        t_rawdat = np.loadtxt(open("../data/{}.txt".format(args.dataset)), delimiter=',')
        if self.localIndex:
            t_rawdat = t_rawdat[:, self.localIndex]
        self.n_sample, self.n0_reality = t_rawdat.shape  # n_sample, n_group
        print(self.n_sample, self.n0_reality)  # 785, 10


        #t_rawdat_mean = np.mean(t_rawdat, axis=1).reshape(-1,1)
        if False:#args.leaderNode:
            self.rawdat = np.concatenate((t_rawdat, t_rawdat_mean), axis=1) #[785,11]
            self.n0 = self.n0_reality + 1  # 加上虚拟节点后的节点数量
        else:
            self.rawdat = t_rawdat
            self.n0 = self.n0_reality


        self.global_nei_num = args.global_nei_num


        print('data shape', self.rawdat.shape)  # 785, 10
        if args.sim_mat:
            #self.load_sim_mat(args)
            self.adj = torch.ones((self.n0+self.global_nei_num, self.n0+self.global_nei_num), device = self.device)
            adj_total = torch.Tensor(np.loadtxt(open("../data/{}.txt".format(args.sim_mat)), delimiter=','))
            self.adj[:self.n0_reality, :self.n0_reality] = adj_total[self.localIndex, :][:,self.localIndex]
            self.degree_adj = torch.sum(self.adj, dim=-1)

            sp_adj = sp.coo_matrix(self.adj)
            self.edge_index = torch.LongTensor(np.vstack((sp_adj.row, sp_adj.col)))

        # if args.extra:
        #     self.load_external(args)

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
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy()  # 199, 47
        self.max = np.max(train_mx, 0)#[:self.n0_reality]
        self.min = np.min(train_mx, 0)#[:self.n0_reality]
        # np.save('%s/maxvalue.npy' % (self.save_dir), self.max)
        # np.save('%s/minvalue.npy' % (self.save_dir), self.min)
        self.peak_thold = np.mean(train_mx, 0)[:self.n0_reality]
        self.dat = (self.rawdat - self.min) / (self.max - self.min + 1e-12)
        print("_pre_train:", self.dat.shape)

    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h)  # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if (train == valid):
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