from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt
import warnings
import scipy.sparse as sp
from scipy.stats.stats import pearsonr
from model.PandemicGNN import *
from utils.data_utils import TrustGNNLoader
from utils.utils import *
from utils.file_utils import *
#from data import *
import torch
import torch.nn.functional as F
import stat
import shutil
import logging
import glob
import time
import yaml

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
    args.dataset = 'state360' #['japan','region785', 'state360']
    args.lr = 0.01
    args.horizon = 10
    institueion_idx = 2 #选择机构，None表示global视图(但是这边无需为global生成嵌入）
    dir_name = args.dataset  # 文件夹名称, 为不同数据集用不同的文件夹
    """以上参数需要修改"""

    # args.dataset: adjacency matrix filename (*-adj.txt)
    if args.dataset == 'state360':
        args.sim_mat = 'state-adj-49'
        institueion_list = [[1, 2, 4, 5, 9, 10, 14, 21, 24, 25, 26, 29, 32, 35, 39, 42, 45, 48], [0, 3, 8, 11, 12, 13, 15, 16, 18, 20, 22, 23, 31, 33, 34, 38, 40, 41, 44, 46, 47], [6, 7, 17, 19, 27, 28, 30, 36, 37, 43]]
    elif args.dataset == 'region785':
        args.sim_mat = 'region-adj'
        institueion_list = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9]]  # 机构划分列表
    elif args.dataset == 'japan':
        args.sim_mat = 'japan-adj'
        institueion_list = [[0, 1, 2, 4, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 46], [3, 5, 10, 19, 23, 45], [7, 11, 24, 25, 31, 38, 44]]
    elif args.dataset == 'spain':
        args.sim_mat = 'spain-adj'
        institueion_list = [[1, 2, 3, 4, 7, 9, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 31, 32, 34, 35, 43, 44, 45, 46, 47, 48, 51], [0, 5, 6, 8, 11, 14, 27, 28, 30, 36, 37, 39, 40, 42, 49, 50], [10, 16, 33, 38, 41]]
    elif args.dataset == 'australia':
        args.sim_mat = 'australia-adj'
        institueion_list = [[0, 1, 6], [2, 4, 7], [3, 5]]


    if institueion_idx is None:
        args.institution_name = args.dataset + '-global'
        args.localIndex = None
    else:
        args.institution_name = args.dataset + '-sub' + chr(ord('A') + institueion_idx)
        args.localIndex = institueion_list[institueion_idx]


    args.runTag = -1  # -1说明是初始生成轮,该文件中必须为-1
    args.global_nei_num = 0  # 邻居数量，该文件中必须为0
    args.hidR = args.k * 4 * args.hidP + args.k #不能修改

    # 共享embedding存储路径
    args.trustGNNpath = os.path.join('..', 'share_emb', dir_name)
    if not os.path.exists(args.trustGNNpath): os.makedirs(args.trustGNNpath)
    # 本地日志路径
    args.local_log_path = os.path.join('.', 'local_log', dir_name)
    if not os.path.exists(args.local_log_path): os.makedirs(args.local_log_path)
    # 存储模型的路径
    if not os.path.exists(os.path.join(args.local_log_path, 'model')):
        os.makedirs(os.path.join(args.local_log_path, 'model'))
    args.temp_model_dir = os.path.join(args.local_log_path, 'model', '{}_runTag{}_tempModel.pt'.format(args.institution_name, args.runTag))
    args.best_model_dir = os.path.join(args.local_log_path, 'model', '{}_runTag{}_bestModel.pt'.format(args.institution_name, args.runTag))



    print('--------Parameters--------')
    print(args)
    print('--------------------------')
    return args

def evaluate(data_loader, data):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    x_value_mx = []
    for [X, Y, index] in data_loader.get_batches(data, batch_size, False):
        output = model(X)[:, :data_loader.n0_reality]
        X = X[:, :, :data_loader.n0_reality]
        Y = Y[:, :data_loader.n0_reality]

        loss_train = F.mse_loss(output, Y)  # mse_loss
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.n0_reality);

        x_value_mx.append(X.data.cpu())
        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    x_value_mx = torch.cat(x_value_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx)  # [n_samples, 47]
    x_value_states = x_value_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[:data_loader.n0_reality]) * 1.0 + data_loader.min[:data_loader.n0_reality]
    y_true_states = y_true_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[:data_loader.n0_reality]) * 1.0 + data_loader.min[:data_loader.n0_reality]
    y_pred_states = y_pred_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[:data_loader.n0_reality]) * 1.0 + data_loader.min[:data_loader.n0_reality]  # (#n_samples, 47)
    rmse_states = np.mean(
        np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))  # mean of 47
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)  # Standard deviation of MAEs for all states/places


    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states, (-1))
    y_pred = np.reshape(y_pred_states, (-1))
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 0.00001)))
    mape /= 10000000
    if not args.pcc:
        pcc = pearsonr(y_true, y_pred)[0]
    else:
        pcc = 1
        pcc_states = 1
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')  # variance_weighted
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples),  rmse, pcc, y_true, y_pred


def train(data_loader, data):
    model.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for [X, Y, index] in data_loader.get_batches(data, batch_size, True):
        # data[0][index,:,:]==X
        # data_loader.totalData[0][index,:,:]==X
        optimizer.zero_grad()
        output = model(X)[:, :data_loader.n0_reality]
        Y = Y[:, :data_loader.n0_reality]
        if Y.size(0) == 1:
            Y = Y.view(-1)
        loss_train = F.mse_loss(output, Y)  # mse_loss
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        n_samples += (output.size(0) * data_loader.n0_reality)
    return float(total_loss / n_samples)

if __name__ == '__main__':
    args = getArgs()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    device = torch.device(args.device)



    data_loader = TrustGNNLoader(args)

    model = PandemicGNN_GAT(args, data_loader)
    print('model %s', model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)



    bad_counter = 0
    best_epoch, best_val = 0, 1e+16
    print('begin training')

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train)
        val_loss, rmse, pcc, _, _ = evaluate(data_loader, data_loader.val)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch,
                                                                                     (time.time() - epoch_start_time),
                                                                                     train_loss, val_loss))
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0

            with open(args.temp_model_dir, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('Best validation epoch:', epoch, time.ctime())
            test_loss, rmse, pcc, _, _ = evaluate(data_loader, data_loader.test)
            print('TEST RMSE {:5.4f}, PCC {:5.4f} '.format(rmse, pcc))

        else:
            bad_counter += 1
            if bad_counter >= args.patience:
                break

    # Load the best saved model.

    with open(args.temp_model_dir, 'rb') as f:
        model.load_state_dict(torch.load(f))
    os.remove(args.temp_model_dir)

    test_loss, rmse, pcc, y_true, y_pred = evaluate(data_loader, data_loader.test)
    print('@args: lr={}, instit={}, bet_val={}, runTag={}.'.format(args.lr, args.institution_name, best_val, args.runTag))
    print('@Final evaluation: TEST RMSE {:5.4f}, PCC {:5.4f} '.format(rmse, pcc))


    # 用最优模型计算所有节点的嵌入
    shouldUpdate = False
    if not os.path.exists(get_log_path(args)):
        shouldUpdate = True
        print("First time to run!")
    else:
        with open(get_log_path(args), "r") as f:
            rel_log = yaml.safe_load(f)
        if best_val <= rel_log['val_loss']:
            shouldUpdate = True
            print("Update embedding!")
        else:
            print("No need to update!")
    print(model(data_loader.totalData[0], saveEmb=True))
    if shouldUpdate:
        # 更新最佳模型
        with open(args.best_model_dir, 'wb') as f:
            torch.save(model.state_dict(), f)
        # 更新嵌入（无需存center-GNN的嵌入）
        if args.localIndex is not None:
            local_feat_emb = model(data_loader.totalData[0], saveEmb=True)
            torch.save(local_feat_emb, get_emb_path(args))
        # 更新log
        rel_log = {'val_loss': best_val,
                   'test_loss': test_loss,
                   'y_true': y_true.tolist(),
                   'y_pred': y_pred.tolist(),
                   'rmse': rmse,
                   'pcc': pcc.tolist(),
                   'args': vars(args)}
        with open(get_log_path(args), "w") as f:
            yaml.dump(rel_log, f)
def get_local_emb(temp_args):
    global args
    global model
    global optimizer
    args = temp_args
    # args.dataset: adjacency matrix filename (*-adj.txt)
    if args.dataset == 'state360':
        args.sim_mat = 'state-adj-49'
        institueion_list = [[1, 2, 4, 5, 9, 10, 14, 21, 24, 25, 26, 29, 32, 35, 39, 42, 45, 48], [0, 3, 8, 11, 12, 13, 15, 16, 18, 20, 22, 23, 31, 33, 34, 38, 40, 41, 44, 46, 47], [6, 7, 17, 19, 27, 28, 30, 36, 37, 43]]
    elif args.dataset == 'region785':
        args.sim_mat = 'region-adj'
        institueion_list = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9]]  # 机构划分列表
    elif args.dataset == 'japan':
        args.sim_mat = 'japan-adj'
        institueion_list = [[0, 1, 2, 4, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 46], [3, 5, 10, 19, 23, 45], [7, 11, 24, 25, 31, 38, 44]]
    elif args.dataset == 'spain':
        args.sim_mat = 'spain-adj'
        institueion_list = [[1, 2, 3, 4, 7, 9, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 31, 32, 34, 35, 43, 44, 45, 46, 47, 48, 51], [0, 5, 6, 8, 11, 14, 27, 28, 30, 36, 37, 39, 40, 42, 49, 50], [10, 16, 33, 38, 41]]
    elif args.dataset == 'australia':
        args.sim_mat = 'australia-adj'
        institueion_list = [[0, 1, 6], [2, 4, 7], [3, 5]]
    institueion_idx = 1#机构1
    args.institution_name = args.dataset + '-sub' + chr(ord('A') + institueion_idx)
    args.localIndex = institueion_list[institueion_idx]
    #mkdir 
    args.trustGNNpath = os.path.join('..', 'share_emb', args.dataset)
    if not os.path.exists(args.trustGNNpath): os.makedirs(args.trustGNNpath)
    args.local_log_path = os.path.join('.', 'local_log', args.dataset)
    if not os.path.exists(args.local_log_path): os.makedirs(args.local_log_path)
    # 存储模型的路径
    if not os.path.exists(os.path.join(args.local_log_path, 'model')):
        os.makedirs(os.path.join(args.local_log_path, 'model'))
    #
    args.temp_model_dir = os.path.join(args.local_log_path, 'model', '{}_runTag{}_tempModel.pt'.format(args.institution_name, args.runTag))
    args.best_model_dir = os.path.join(args.local_log_path, 'model', '{}_runTag{}_bestModel.pt'.format(args.institution_name, args.runTag))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    device = torch.device(args.device)



    data_loader = TrustGNNLoader(args)
    
    model = PandemicGNN_GAT(args, data_loader)
    print('model %s', model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)



    bad_counter = 0
    best_epoch, best_val = 0, 1e+16
    print('begin training')

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train)
        val_loss, rmse, pcc, _, _ = evaluate(data_loader, data_loader.val)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch,
                                                                                     (time.time() - epoch_start_time),
                                                                                     train_loss, val_loss))
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0

            with open(args.temp_model_dir, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('Best validation epoch:', epoch, time.ctime())
            test_loss, rmse, pcc, _, _ = evaluate(data_loader, data_loader.test)
            print('TEST RMSE {:5.4f}, PCC {:5.4f} '.format(rmse, pcc))

        else:
            bad_counter += 1
            if bad_counter >= args.patience:
                break

    # Load the best saved model.

    with open(args.temp_model_dir, 'rb') as f:
        model.load_state_dict(torch.load(f))
    os.remove(args.temp_model_dir)

    test_loss, rmse, pcc, y_true, y_pred = evaluate(data_loader, data_loader.test)
    print('@args: lr={}, instit={}, bet_val={}, runTag={}.'.format(args.lr, args.institution_name, best_val, args.runTag))
    print('@Final evaluation: TEST RMSE {:5.4f}, PCC {:5.4f} '.format(rmse, pcc))


    # 用最优模型计算所有节点的嵌入
    shouldUpdate = False
    if not os.path.exists(get_log_path(args)):
        shouldUpdate = True
        print("First time to run!")
    else:
        with open(get_log_path(args), "r") as f:
            rel_log = yaml.safe_load(f)
        if best_val <= rel_log['val_loss']:
            shouldUpdate = True
            print("Update embedding!")
        else:
            print("No need to update!")
    
    if shouldUpdate:
        # 更新最佳模型
        with open(args.best_model_dir, 'wb') as f:
            torch.save(model.state_dict(), f)
        # 更新嵌入（无需存center-GNN的嵌入）
        if args.localIndex is not None:
            local_feat_emb = model(data_loader.totalData[0], saveEmb=True)
            torch.save(local_feat_emb, get_emb_path(args))
        # 更新log
        rel_log = {'val_loss': best_val,
                   'test_loss': test_loss,
                   'y_true': y_true.tolist(),
                   'y_pred': y_pred.tolist(),
                   'rmse': rmse,
                   'pcc': pcc.tolist(),
                   'args': vars(args)}
        with open(get_log_path(args), "w") as f:
            yaml.dump(rel_log, f)
    return model(data_loader.totalData[0], saveEmb=True)