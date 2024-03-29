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
from utils.data_utils import *
from utils.utils import *
from utils.file_utils import *
import torch
import torch.nn.functional as F
import stat
import shutil
import logging
import glob
import time
#from tensorboardX import SummaryWriter
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
    return args

def updateArgs(args):
    """
    参数更新（因为有些参数依赖于已有参数）
    """
    # 获取机构划分
    args.sim_mat, _ = gen_institution(args.dataset, args.clientNum)
    args.institution_name, args.localIndex = get_subInstitution(args.dataset,args.clientNum, args.institueion_idx)

    args.global_nei_num = 1
    args.hidR = args.k * 4 * args.hidP + args.k  # 不能修改

    gen_CNL_dir(args.dir_name)
    args.temp_model_dir = get_temp_model_dir(args)
    args.best_model_dir = get_best_model_dir(args)
    print('--------Parameters--------')
    print(args)
    print('--------------------------')
    return args




def runExp(args):

    def evaluate(data_loader, data, share_emb):
        model.eval()
        total = 0.
        n_samples = 0.
        total_loss = 0.
        batch_size = args.batch
        y_pred_mx = []
        y_true_mx = []
        x_value_mx = []
        for [X, Y, index] in data_loader.get_batches(data, batch_size, False):
            # output = model(X, trustGNNpath=args.trustGNNpath, index = index)[:, :data_loader.n0_reality]
            output = model(X, shareEmb=share_emb[index, :, :].detach().to(device))
            output = output[:, :data_loader.n0_reality]
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
        x_value_states = x_value_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[
                                                                                          :data_loader.n0_reality]) * 1.0 + data_loader.min[
                                                                                                                            :data_loader.n0_reality]
        y_true_states = y_true_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[
                                                                                        :data_loader.n0_reality]) * 1.0 + data_loader.min[
                                                                                                                          :data_loader.n0_reality]
        y_pred_states = y_pred_mx.numpy() * (data_loader.max[:data_loader.n0_reality] - data_loader.min[
                                                                                        :data_loader.n0_reality]) * 1.0 + data_loader.min[
                                                                                                                          :data_loader.n0_reality]  # (#n_samples, 47)
        rmse_states = np.mean(
            np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))  # mean of 47
        raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')

        # convert y_true & y_pred to real data
        y_true = np.reshape(y_true_states, (-1))
        y_pred = np.reshape(y_pred_states, (-1))
        metrics = regression_metrics(y_true, y_pred)

        return float(total_loss / n_samples), metrics, y_true_states, y_pred_states

    def train(data_loader, data, share_emb):
        model.train()
        total_loss = 0.
        n_samples = 0.
        batch_size = args.batch

        for [X, Y, index] in data_loader.get_batches(data, batch_size, True):
            # data[0][index,:,:]==X
            # data_loader.totalData[0][index,:,:]==X
            optimizer.zero_grad()

            output = model(X, shareEmb=share_emb[index, :, :].detach().to(device))
            output = output[:, :data_loader.n0_reality]
            Y = Y[:, :data_loader.n0_reality]
            if Y.size(0) == 1:
                Y = Y.view(-1)
            loss_train = F.mse_loss(output, Y)  # mse_loss
            total_loss += loss_train.item()
            loss_train.backward()
            optimizer.step()
            n_samples += (output.size(0) * data_loader.n0_reality)
        return float(total_loss / n_samples)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    data_loader = TrustGNNLoader(args)

    # 从文件夹中获取邻居的嵌入
    share_emb = get_shareEmb(args, n_sample=data_loader.totalData[0].shape[0])

    net = get_net(args.model)
    model = net(args, data_loader).to(device)
    print('model %s', model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)

    bad_counter = 0
    best_epoch, best_val = 0, 1e+16
    print('begin training')

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train, share_emb)
        val_loss, _, _, _ = evaluate(data_loader, data_loader.val, share_emb=share_emb)
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
            print('Best validation epoch:', epoch, time.ctime());
            test_loss, metrics, _, _ = evaluate(data_loader, data_loader.test, share_emb=share_emb)
            print('TEST RMSE {:5.4f}, PCC {:5.4f} '.format(metrics['rmse'], metrics['pcc']))

        else:
            bad_counter += 1
            if bad_counter >= args.patience:
                break

    # Load the best saved model.
    with open(args.temp_model_dir, 'rb') as f:
        model.load_state_dict(torch.load(f))
    os.remove(args.temp_model_dir)

    test_loss, metrics, y_true_states, y_pred_states = evaluate(data_loader, data_loader.test, share_emb=share_emb)

    print(
        '@args: lr={}, instit={}, bet_val={}, runTag={}.'.format(args.lr, args.institution_name, best_val, args.runTag))
    print('@Final evaluation: TEST RMSE {:5.4f}, PCC {:5.4f} '.format(metrics['rmse'], metrics['pcc']))

    # 用最优模型计算所有节点的嵌入
    if args.localIndex is not None:
        # 判定当前runTag是否存在log.yaml，若不存在则读取上一轮的log.yaml
        shouldUpdate = False
        if os.path.exists(get_log_path(args)):
            with open(get_log_path(args), "r") as f:
                rel_log = yaml.safe_load(f)
            if best_val >= rel_log['val_loss']:
                print("Performance deteriorates and thus is not updated! TAT")
        else:
            with open(get_log_path(args, runTag=args.runTag - 1), "r") as f:
                rel_log = yaml.safe_load(f)
            # 模型性能变差 -> 复制上一轮的信息
            if best_val > rel_log['val_loss']:
                shouldUpdate = True
                local_feat_emb = torch.load(get_shareEmb_path(args, args.runTag - 1))
                with open(get_best_model_dir(args, runTag=args.runTag - 1), 'rb') as f:
                    model.load_state_dict(torch.load(f))
                print("Duplicate old version embedding! =_=")

        # 模型性能提升->更新emb和log
        if best_val <= rel_log['val_loss']:
            shouldUpdate = True
            local_feat_emb = model(data_loader.totalData[0], saveEmb=True)
            rel_log = {'val_loss': best_val,
                       'test_loss': test_loss,
                       'y_true_states': y_true_states.tolist(),
                       'y_pred_states': y_pred_states.tolist(),
                       'args': vars(args)}
            rel_log.update(metrics)
            print("Better performance, update embedding! ^_^")

        if shouldUpdate:
            # 更新最佳模型
            with open(args.best_model_dir, 'wb') as f:
                torch.save(model.state_dict(), f)
            # 更新嵌入(.pt文件)
            torch.save(local_feat_emb, get_shareEmb_path(args))
            # 更新yaml文件
            with open(get_log_path(args), "w") as f:
                yaml.dump(rel_log, f)

if __name__ == '__main__':
    args = getArgs()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    """以下参数需要修改"""
    args.dataset = 'twitter'  # ['japan','region785', 'state360', 'china']
    args.model = 'GAT'
    args.horizon = 7
    args.clientNum = 3
    runTag = 0  # 读取的emb的runTag（从0开始逐渐增加，若没有机构获得性能提升则停止）
    args.dir_name = '{}_{}_w20h{}c{}'.format(args.model, args.dataset, args.horizon, args.clientNum)# 文件夹名称, 为不同数据集用不同的文件夹
    """以上参数需要修改"""
    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
    institueion_idx_list = list(range(args.clientNum)) #待遍历的机构列表
    for lr, institueion_idx in itertools.product(lr_list, institueion_idx_list):
        args.lr = lr
        args.institueion_idx = institueion_idx
        args.runTag = runTag
        args = updateArgs(args)
        runExp(args)

    print("Finish! ^_^")







