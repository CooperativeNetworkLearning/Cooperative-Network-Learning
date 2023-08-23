from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
from data_loader import *
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
from models import get_net
from utils.data_utils import DataLoader, CNL_DataLoader, random_planetoid_splits, gen_institution, gen_institution_mask

#from data import *
import torch
import torch.nn.functional as F
import stat
import shutil
import logging
import glob
import time
import yaml


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

def get_local_emb(temp_args):
    global args
    global model,device
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
    institueion_idx = 1#机构2
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

    print(args)
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
    if not os.path.exists(get_log_path_node(args)):
        shouldUpdate = True
        print("First time to run!")
    else:
        with open(get_log_path_node(args), "r") as f:
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
        with open(get_log_path_node(args), "w") as f:
            yaml.dump(rel_log, f)
    return model(data_loader.totalData[0], saveEmb=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def RunExp_local(args, dataset, data, Net, percls_trn, val_lb,  institution_mask):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gnn_net = Net(dataset, args)


    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)
    
    model, data = gnn_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        del out

    def test():
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    def evaluate_sub_institution(model, data, institution_mask):
        """
        评估测试集中，不同机构的性能
        """
        model.eval()
        test_mask = data.test_mask
        logits = model(data)
        acc=[]
        for sub_ins_mask in institution_mask:
            mask = torch.bitwise_and(test_mask, sub_ins_mask)
            pred = logits[mask].max(1)[1]
            sub_acc =  pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            acc.append(sub_acc)

        #logits[test_mask].max(1)[1].eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
        return acc



    

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history, val_acc_history = [], []
    for epoch in range(args.epochs):
        
        train()

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test()

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch >= 1000:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break


    model.load_state_dict(best_model_wts)
    print(f'seed{args.seed} @ institueion_idx:{args.institueion_idx}, lr:{args.lr}, val_loss:{best_val_loss.item()}, test_acc:{test_acc}, val_acc:{best_val_acc}')

    # 用最优模型计算所有节点的嵌入
    if True:
        # 更新嵌入（无需存center-GNN的嵌入）
        if args.localIndex is not None:
            local_feat_emb = model(data, saveEmb=True)
            torch.save(local_feat_emb, get_shareEmb_path(args))

        # 更新log
        args.localIndex = len(args.localIndex) if args.localIndex is not None else None
        rel_log = {'val_loss': best_val_loss.item(),
                   'val_acc': best_val_acc,
                   'test_acc': test_acc,
                   'args': vars(args)}
        if args.localIndex is None:
            rel_log['sub_acc'] = evaluate_sub_institution(model,data, institution_mask)
        with open(get_log_path(args), "w") as f:
            yaml.dump(rel_log, f)
    return local_feat_emb

def get_node_emb(args):
    Net = get_net(args.net)
    dataset, data = DataLoader(args.dataset, args)
    args.dir_name = f'{args.dataset}_{args.net}_c{args.clientNum}_seed{args.seed}' # 文件夹名称, 为不同数据集用不同的文件夹
    institution_mask = gen_institution_mask(args.dataset, args.clientNum, data.num_nodes).to(device)
    gen_CNL_dir(args.dir_name)
    args.institution_name, args.localIndex = gen_institution(args.dataset, args.clientNum, args.institueion_idx)
    dataset, data = CNL_DataLoader(args.dataset, args)
    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))
    return RunExp_local(args, dataset, data, Net, percls_trn, val_lb, institution_mask)

def get_local_emb_switch(args):

    if args.tasktype == 0:
        return get_local_emb(args)
    if args.tasktype == 1:
        return get_node_emb(args)
    if args.tasktype == 2:
        return get_ciao_emb(args)
    
    


##edge 
from fedml_subgraph_link_prediction import create_model
import pickle

def get_ciao_emb(args):
    ## 如果数据不存在，则生成数据
    if not os.path.exists('./data/ciao/graphs.pk'):
        parser = argparse.ArgumentParser(description='test')
        ap = parser.parse_args([])
        ap.client_num_in_total = 3
        setup_seed(1)
        Gs = get_data_category(args, './data/', 'ciao', load_processed=False)
        with open('./data/ciao/graphs.pk', 'wb') as f:
            pickle.dump(Gs, f)
    with open('./data/ciao/graphs.pk', 'rb') as f:
        Gs = pickle.load(f)
        data = Gs[args.institueion_idx]
    setup_seed(args.seed)
    #定义模型
    train_data = data
    metric_fn = mean_absolute_error
    device = 'cuda:%s'%str(args.device)
    model = create_model(model_name = args.model_name, feature_dim = train_data.x.shape[1], node_embedding_dim=32,hidden_size=63,num_heads=8)
    model = model.to(device)
    train_data = train_data.to(device)
    
    def test(test_data, device, val=True, metric=mean_absolute_error,if_center=False):
        model.eval()
        model.to(device)
        metric = metric
        mae, rmse, mse = [], [], []

        batch = test_data.to(device)
        num_dim = 1169
        if batch.x.shape[1] > num_dim:
            batch.x = batch.x[:,:num_dim]
        elif batch.x.shape[1] < num_dim:
            batch.x = torch.concat((batch.x,torch.zeros(batch.x.size(0),num_dim-batch.x.size(1)).to(device)),dim=1)
        with torch.no_grad():
            train_z = model.encode(batch.x, batch.edge_train)
            if val:
                link_logits = model.decode(train_z, batch.edge_val)
            else:
                link_logits = model.decode(train_z, batch.edge_test)

            if val:
                link_labels = batch.label_val
            else:
                link_labels = batch.label_test
            score = metric(link_labels.cpu(), link_logits.cpu())
            mae.append(mean_absolute_error(link_labels.cpu(), link_logits.cpu()))
            rmse.append(
                mean_squared_error(
                    link_labels.cpu(), link_logits.cpu(), squared=False
                )
            )
            mse.append(mean_squared_error(link_labels.cpu(), link_logits.cpu()))
        return score, model, mae, rmse, mse
    def train(train_data, device, args):
        model.to(device)
        model.train()
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        max_test_score = np.inf
        best_model_params = {}
        for epoch in range(args.epochs):
            batch = train_data
            batch.to(device)
            optimizer.zero_grad()

            z = model.encode(batch.x, batch.edge_train)
            link_logits = model.decode(z, batch.edge_train)
            link_labels = batch.label_train
            loss = F.mse_loss(link_logits, link_labels)
            loss.backward()
            optimizer.step()
            if epoch % 10==0:
                test_score, _, _, _, _ = test(
                    train_data, device, val=True, metric=metric_fn
                )
                if test_score<max_test_score:
                    max_test_score = test_score
                    best_model_params = {
                        k: v.cpu() for k, v in model.state_dict().items()
                    }
        return max_test_score, best_model_params
    
    _,best_model_params = train(train_data, device, args)
    model.load_state_dict(best_model_params)
    #save best embedding
    z = model.encode(train_data.x, train_data.edge_train)
    embed = torch.mean(z, dim=0).unsqueeze(0)
    return embed
    