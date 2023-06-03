from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp
from scipy.stats.stats import pearsonr
from model.PandemicGNN import *
from utils.data_utils import TrustGNNLoader, gen_institueion
from utils.utils import *
from data import *
from utils.file_utils import *
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

"""
该文件为画时序预测图进行数据准备（会用存下来的最佳模型）
运行顺序：
(1) runTag=-1, institueion_idx = None       会生成 xx_centralized.vsv 和 xx_gt.csv
(2) runTag=-1, institution_idx = 0, 1, 2... 会生成 xx_local.csv
(3) runTag>-1, institution_idx = 0, 1, 2... 会生成 xx_integrated.csv
注意：institution_idx必须严格按顺序执行，（因为需要把数据依次拼起来）
"""

def getArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='region785', help="Dataset string")
    ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)")
    ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)")
    ap.add_argument('--seed', type=int, default=42, help='random seed')
    ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
    ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
    ap.add_argument('--batch', type=int, default=128, help="batch size")
    ap.add_argument('--check_point', type=int, default=1, help="check point")
    ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
    ap.add_argument('--train', type=float, default=.5, help="Training ratio (0, 1)")
    ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
    ap.add_argument('--test', type=float, default=.3, help="Testing ratio (0, 1)")
    ap.add_argument('--mylog', action='store_false', default=True, help='save tensorboad log')
    ap.add_argument('--cuda', action='store_true', default=False, help='')
    ap.add_argument('--window', type=int, default=20, help='')
    ap.add_argument('--horizon', type=int, default=5, help='leadtime default 5')
    ap.add_argument('--model_dir', type=str, default='save', help='dir path to save the final model')
    ap.add_argument('--gpu', type=int, default=0, help='choose gpu 0-10')
    ap.add_argument('--lamda', type=float, default=0.01, help='regularize params similarities of states')
    ap.add_argument('--patience', type=int, default=50, help='patience default 100')
    ap.add_argument('--k', type=int, default=8, help='kernels')
    ap.add_argument('--hidA', type=int, default=64, help='hidden dim of attention layer')
    ap.add_argument('--hidP', type=int, default=1, help='hidden dim of adaptive pooling')
    ap.add_argument('--extra', type=str, default='', help='externel folder')
    ap.add_argument('--label', type=str, default='', help='label_file')
    ap.add_argument('--pcc', type=str, default='', help='have pcc?')
    ap.add_argument('--n', type=int, default=2, help='layer number of GCN')
    ap.add_argument('--res', type=int, default=0, help='0 means no residual link while 1 means need residual link')
    ap.add_argument('--s', type=int, default=2, help='kernel size of temporal convolution network')
    ap.add_argument('--result', type=int, default=0, help='0 means do not show result while 1 means show result')
    ap.add_argument('--ablation', type=str, default=None, help='ablation test')
    ap.add_argument('--eval', type=str, default='', help='evaluation test file')
    ap.add_argument('--record', type=str, default='', help='record the result')
    ap.add_argument('--model', type=str, default='GAT', help='model')
    args = ap.parse_args()

    """以下参数需要修改"""
    args.dataset = 'state360' #['japan','region785', 'state360', 'china']
    args.horizon = 10
    args.runTag = 0  #
    args.institueion_idx = 2 #选择机构，None表示global视图(但是这边无需为global生成嵌入）
    args.dir_name = 'state360_w20h10_client3'  # 文件夹名称, 为不同数据集用不同的文件夹
    """以上参数需要修改"""

    # 获取机构划分
    args.sim_mat, institueion_list = gen_institueion(args.dataset)
    args.institution_name, args.localIndex = get_subInstitution(args.dataset, args.institueion_idx)
    if args.runTag == -1:
        args.global_nei_num = 0
    elif args.runTag > -1:
        args.global_nei_num = 1
    args.hidR = args.k * 4 * args.hidP + args.k  # 不能修改

    gen_CNL_dir(args.dir_name)
    args.temp_model_dir = get_temp_model_dir(args)
    args.best_model_dir = get_best_model_dir(args)

    print('--------Parameters--------')
    print(args)
    print('--------------------------')
    return args

def get_gt_path(args):
    param_ = '_w{}h{}_'.format(args.window, args.horizon)
    return os.path.join('..', 'pic', 'plot_data', args.dataset + param_+ '_gt.csv')

def get_timeSeries_store_path(args):
    param_ = '_w{}h{}_'.format(args.window, args.horizon)
    if args.runTag>-1:
        tag_ = '_integrated'
    elif args.runTag==-1 and args.localIndex is None:
        tag_ = '_centralized'
    elif args.runTag==-1 and args.localIndex is not None:
        tag_ = '_local'
    return os.path.join('..', 'pic','plot_data', args.dataset +  param_+  tag_ + '.csv')


if __name__ == '__main__':
    args = getArgs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    device = torch.device(args.device)
    data_loader = TrustGNNLoader(args)
    model = PandemicGNN_GAT(args, data_loader)

    # Load the best saved model.
    with open(args.best_model_dir, 'rb') as f:
        model.load_state_dict(torch.load(f))

    if args.runTag > -1:
        share_emb = get_shareEmb(args, n_sample=data_loader.totalData[0].shape[0])

    model.eval()
    batch_size = args.batch
    y_pred_mx, y_true_mx = [], []
    splitLen = []
    for data in [data_loader.train, data_loader.val, data_loader.test]:
        splitLen.append(len(data[0]))
        for [X, Y, index] in data_loader.get_batches(data, batch_size, False):
            if args.runTag > -1:
                output = model(X, shareEmb=share_emb[index, :, :].detach().to(device))
            else:
                output = model(X)

            output = output[:, :data_loader.n0_reality]
            X = X[:, :, :data_loader.n0_reality]
            Y = Y[:, :data_loader.n0_reality]

            y_true_mx.append(Y.data.cpu())
            y_pred_mx.append(output.data.cpu())

    print('size of train/val/test sets', splitLen)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx)
    y_true_states = y_true_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[:data_loader.n0_reality]) * 1.0 + data_loader.min[:data_loader.n0_reality]
    y_pred_states = y_pred_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[:data_loader.n0_reality]) * 1.0 + data_loader.min[:data_loader.n0_reality]  # (#n_samples, 47)

    test_rmse = sqrt(mean_squared_error(np.reshape(y_true_states[-splitLen[-1]:y_true_states.shape[0],:], (-1)), np.reshape(y_pred_states[-splitLen[-1]:y_pred_states.shape[0],:], (-1))))
    print("test_rmse:", test_rmse)

    # 将预测数据写入csv文件用于后续画图
    data_path = get_timeSeries_store_path(args)
    if os.path.exists(data_path) and (args.institueion_idx is not None) and (args.institueion_idx>0):
        t_y_pred_states = pd.read_csv(data_path, header=None).values
        y_pred_states = np.append(t_y_pred_states, y_pred_states, axis=1)

    dataframe = pd.DataFrame(y_pred_states)
    dataframe.to_csv(data_path, mode="w", index=False, header=None)

    if args.localIndex is None and args.runTag == -1: #说明是centralized模型
        gt_path = get_gt_path(args)
        dataframe = pd.DataFrame(y_true_states)
        dataframe.to_csv(gt_path, mode="w", index=False, header=None)

    # 画图
    x = list(range(y_pred_states.shape[0]))
    plt.plot(x, np.mean(y_true_states, axis=1))
    plt.plot(x, np.mean(y_pred_states, axis=1))
    plt.show()


