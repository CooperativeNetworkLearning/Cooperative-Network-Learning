from data_loader import *
import argparse
from fedml_subgraph_link_prediction import create_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import _thread
from tqdm import tqdm
## get data
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--seed', type=int, default=0,
                    help='ramdon seed')

parser.add_argument('--device', type=int, default=-1,
                    help='ramdon seed')
args = parser.parse_args()

args.client_num_in_total = 3
args.metric = "MAE"
args.learning_rate = 0.003
args.weight_decay = 0.001
args.epochs = 5000
args.client_optimizer = "sgd"
args.model_name = 'gcn'
res = {}
res_ins = {}
seed = args.seed
setup_seed(seed)
print('load_graph...and classifiction')
args.client_num_in_total = 3
Gs = get_data_category(args, './', 'ciao', load_processed=False)
args.client_num_in_total = 1
# center_G = get_data_category(args, './', 'ciao', load_processed=False)[0]
device = 'cuda:%s'%str(args.device)
## 获取嵌入
def get_share_emb(ins_num, seed):
    file_temp = './share_emb/Zseed-%s-%s'
    self_emb = torch.load(file_temp%(str(seed),str(ins_num))).to('cpu')
    share_emb = torch.zeros(self_emb.size(), device=self_emb.device)
    n = 0
    for i in range(3):
        if i==ins_num:
            continue
        share_emb += torch.load(file_temp%(str(seed),str(i))).to('cpu')
        n += 1
    return share_emb/n
#逐机构评估-intergrated 
def institution_inter_ev(ins_num,data,seed):
    #定义模型
    train_data = data
    metric_fn = mean_absolute_error
    device = 'cuda:%s'%str(args.device)
    model = create_model(model_name = args.model_name, feature_dim = train_data.x.shape[1], node_embedding_dim=32,hidden_size=63,num_heads=8)
    model = model.to(device)
    train_data = train_data.to(device)
    # 添加虚拟节点和连边，其和机构中所有节点相连
    node_num = train_data.index_orig.size(0)
    new_nodes = list(range(node_num, node_num + 1))
    new_edges = torch.tensor([new_nodes * node_num,
                                [i // len(new_nodes) for i in range(node_num * len(new_nodes))]], dtype=torch.int64).to(device)

    existing_edges = torch.cat([train_data.edge_train, new_edges, new_edges.flip(0)], dim=1)
    train_data.intergrated_edge = existing_edges
    ##new_node 嵌入
    a = get_share_emb(ins_num, seed).detach().to(device)
    new_node_emb = torch.concat((a,torch.zeros(1,Gs[0].x.size(1)-a.size(1)).to(device)),dim=1)
    train_data.x = torch.concat((train_data.x,new_node_emb))
    ##
    def test(test_data, device, val=True, metric=mean_absolute_error,if_center=False):
        model.eval()
        model.to(device)
        metric = metric
        mae, rmse, mse = [], [], []

        batch = test_data
        batch = batch.to(device)
        with torch.no_grad():
            train_z = model.encode(batch.x, batch.intergrated_edge)
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
            if val == False:
                if if_center == False:
                    with open('./result/res_%s'%str(ins_num), 'w+') as f:
                        f.write("{},{},{},{}\n".format(seed, mae, rmse, mse))
                        if ins_num not in res.keys():
                            res[ins_num] = []
                        res[ins_num].append((seed, mae[0], rmse[0], mse[0]))
                else:
                    if ins_num not in res_ins.keys():
                            res_ins[ins_num] = []
                    res_ins[ins_num].append((seed, mae[0], rmse[0], mse[0]))
            
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

            z = model.encode(batch.x, batch.intergrated_edge)
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
    test(train_data, device, val=False, metric=mean_absolute_error)
    if i == -1:
        for j in range(3):
            test(Gs[j], device, val=False, metric=mean_absolute_error, if_center=True)
    #save best embedding
    z = model.encode(train_data.x, train_data.edge_train)
    embed = torch.mean(z, dim=0).unsqueeze(0)
    file_name = './share_emb/Zseed-{}-{}'.format(seed,ins_num)
    torch.save(embed, file_name)
    
    embed = torch.mean(train_data.x, dim=0).unsqueeze(0)
    file_name = './share_emb/Xseed-{}-{}'.format(seed,ins_num)
    torch.save(embed, file_name)
for i in range(0,3):
    institution_inter_ev(i,Gs[i],seed)
import pickle
f = open('./res_path/interg_res_seed%s'%str(seed),'wb')
pickle.dump(res,f)
f.close()
