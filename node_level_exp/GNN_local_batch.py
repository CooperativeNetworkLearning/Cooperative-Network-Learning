import argparse
import itertools
import os

import yaml
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from data_utils import DataLoader, CNL_DataLoader, random_planetoid_splits, gen_institution, gen_institution_mask
from models import get_net
import random
from file_utils import get_log_path, get_temp_model_dir, get_best_model_dir, gen_CNL_dir, get_shareEmb_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='computers', help='dataset name')
    parser.add_argument('--net', type=str, default='GCN')
    parser.add_argument('--RPMAX', type=int, default=10, help='repeat times')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--clientNum', type=int, default=3)
    parser.add_argument('--neighborAgencyNum', type=int, default=0)
    parser.add_argument('--runTag', type=int, default=-1, help='local模型中固定为-1')

    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    return args


def RunExp_local(args, dataset, data, Net, percls_trn, val_lb,  institution_mask):
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
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



    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gnn_net = Net(dataset, args)


    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)

    model, data = gnn_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history, val_acc_history = [], []
    for epoch in range(args.epochs):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            with open(get_temp_model_dir(args), 'wb') as f:
                torch.save(model.state_dict(), f)


        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    # Load the best saved model.
    with open(get_temp_model_dir(args), 'rb') as f:
        model.load_state_dict(torch.load(f))
    os.remove(get_temp_model_dir(args))
    print(f'seed{args.seed} @ institueion_idx:{institueion_idx}, lr:{lr}, val_loss:{best_val_loss.item()}, test_acc:{test_acc}, val_acc:{best_val_acc}')

    # 用最优模型计算所有节点的嵌入
    shouldUpdate = False
    if not os.path.exists(get_log_path(args)):
        shouldUpdate = True
        print("First time to run!")
    else:
        with open(get_log_path(args), "r") as f:
            rel_log = yaml.safe_load(f)
        if best_val_loss <= rel_log['val_loss']:
            shouldUpdate = True
            print("Update embedding!")
        else:
            print("No need to update!")

    if shouldUpdate:
        # 更新最佳模型
        with open(get_best_model_dir(args), 'wb') as f:
            torch.save(model.state_dict(), f)
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




if __name__ == '__main__':
    args =  parse_args()
    Net = get_net(args.net)

    dataset, data = DataLoader(args.dataset, args)
    for RP in tqdm(range(0,args.RPMAX)):
        args.seed = RP
        args.dir_name = f'{args.dataset}_{args.net}_c{args.clientNum}_seed{args.seed}' # 文件夹名称, 为不同数据集用不同的文件夹
        lr_list=[0.001,0.005,0.01,0.05,0.1]
        institueion_idx_list =  list(range(args.clientNum)) + [None]
        institution_mask = gen_institution_mask(args.dataset, args.clientNum, data.num_nodes).to(device)
        for institueion_idx in institueion_idx_list:
            gen_CNL_dir(args.dir_name)
            args.institueion_idx = institueion_idx

            for lr in lr_list:
                args.lr = lr
                args.institution_name, args.localIndex = gen_institution(args.dataset, args.clientNum, args.institueion_idx)
                #dataset, data = DataLoader(args.dataset, args)
                dataset, data = CNL_DataLoader(args.dataset, args)
                percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
                val_lb = int(round(args.val_rate * len(data.y)))
                RunExp_local(args, dataset, data, Net, percls_trn, val_lb, institution_mask)
                # TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
                # print('True Label rate: ', TrueLBrate)




