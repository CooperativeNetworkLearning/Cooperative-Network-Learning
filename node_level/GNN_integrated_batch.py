import argparse
import itertools
import os

import yaml
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from data_utils import DataLoader, CNL_DataLoader, random_planetoid_splits, gen_institution
from models import get_net
import random
from file_utils import get_log_path, get_temp_model_dir, get_best_model_dir, gen_CNL_dir, get_shareEmb_path, get_shareEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='computers', help='dataset name')
    parser.add_argument('--net', type=str, default='GCN')
    parser.add_argument('--RPMAX', type=int, default=3, help='repeat times')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--clientNum', type=int, default=3)
    parser.add_argument('--neighborAgencyNum', type=int, default=1)
    parser.add_argument('--runTag', type=int, default=0, help='integrated模型中从0开始递增')
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


def RunExp_integrated(args, dataset, data, Net, percls_trn, val_lb):
    def train(model, optimizer, data, share_emb):
        model.train()
        optimizer.zero_grad()
        out = model(data, saveEmb=False, shareEmb = share_emb)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        del out

    def test(model, data, share_emb):
        model.eval()
        logits, accs, losses, preds = model(data, saveEmb=False, shareEmb = share_emb), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 从文件夹中获取邻居的嵌入
    share_emb = get_shareEmb(args)

    gnn_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)

    model, data = gnn_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history, val_acc_history = [], []
    for epoch in range(args.epochs):
        train(model, optimizer, data, share_emb)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, share_emb)

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

    assert args.institueion_idx is not None, "Integrated mode should not employed for centralization!"

    shouldUpdate = False
    if os.path.exists(get_log_path(args)):
        with open(get_log_path(args), "r") as f:
            rel_log = yaml.safe_load(f)
        if best_val_loss >= rel_log['val_loss']:
            print("Performance deteriorates and thus is not updated! TAT")
    else:
        with open(get_log_path(args, runTag=args.runTag - 1), "r") as f:
            rel_log = yaml.safe_load(f)
        # 模型性能变差 -> 复制上一轮的信息
        if best_val_loss > rel_log['val_loss']:
            shouldUpdate = True
            local_feat_emb = torch.load(get_shareEmb_path(args, args.runTag - 1))
            with open(get_best_model_dir(args, runTag=args.runTag - 1), 'rb') as f:
                model.load_state_dict(torch.load(f))
            print("Duplicate old version embedding! =_=")

    # 模型性能提升->更新emb和log
    if best_val_loss <= rel_log['val_loss']:
        shouldUpdate = True
        local_feat_emb = model(data, saveEmb=True)
        args.localIndex = len(args.localIndex)
        rel_log = {'val_loss': best_val_loss.item(),
                   'val_acc': best_val_acc,
                   'test_acc': test_acc,
                   'args': vars(args)}
        print("Better performance, update embedding! ^_^")

    if shouldUpdate:
        # 更新最佳模型
        with open(get_best_model_dir(args), 'wb') as f:
            torch.save(model.state_dict(), f)
        # 更新嵌入(.pt文件)
        torch.save(local_feat_emb, get_shareEmb_path(args))
        # 更新yaml文件
        with open(get_log_path(args), "w") as f:
            yaml.dump(rel_log, f)




if __name__ == '__main__':
    args =  parse_args()
    Net = get_net(args.net)


    for RP in tqdm(range(0,10)):
        args.seed = RP
        args.dir_name = f'{args.dataset}_{args.net}_c{args.clientNum}_seed{args.seed}' # 文件夹名称, 为不同数据集用不同的文件夹
        lr_list=[0.001,0.005,0.01,0.05,0.1]
        institueion_idx_list = list(range(args.clientNum))
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
                RunExp_integrated(args, dataset, data, Net, percls_trn, val_lb)
                # TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
                # print('True Label rate: ', TrueLBrate)




