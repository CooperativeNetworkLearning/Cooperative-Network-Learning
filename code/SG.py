from json import load
from mimetypes import init
import zerorpc
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import pickle
import configparser
##嵌入模型
import sys
sys.path.append('./model')
import rawGNN
model_dict={
    'rawGNN':rawGNN.GCN
}


#假设这个机构的数据 是 TUDataset 的前50个图
local_data ={
    'TU' : TUDataset(root='d:/data/TUDataset', name='MUTAG')[0:50]
}
dataset = TUDataset(root='d:/data/TUDataset', name='MUTAG')[0:50]


class SG(object):
    @zerorpc.stream
    def pullGraphEmbed(self,model_name, data_name):
        # _model = pickle.loads(model)
        _model = model_dict[model_name](local_data[data_name].num_features, 32, local_data[data_name].num_classes)
        loader = DataLoader(local_data[data_name], batch_size=len(local_data[data_name]), shuffle=True)
        # x = torch.empty()
        # y = torch.empty()
        for data in loader:
            embed_x = _model.embed(data.x, data.edge_index, data.batch)
            # x = torch.concat([x,embed_x],axis = 0)
            # y = torch.concat([y,data.y],axis = 0)
            x = embed_x
            y = data.y
        print('处理完毕')
        return pickle.dumps(x), pickle.dumps(y)
##注册报告自身
from SN import get_SN_peer
get_SN_peer()

config = configparser.ConfigParser()
config.read('./config.ini',encoding='utf-8')
s = zerorpc.Server(SG())

s.bind("tcp://0.0.0.0:%s"%config.get('SG', 'port'))
s.run()

# class SGexcute(object):
#     def __init__(self, model, data_name) -> None:
#         #当前版本:模型是远程的，数据是本地的
#         self.model = model
#         self.data = data_name
        
#     def get_local_graph_data(data_name):
        