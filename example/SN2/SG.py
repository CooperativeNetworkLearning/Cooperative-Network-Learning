from json import load
from mimetypes import init
import numpy as np
import zerorpc
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import pickle
import configparser
from torch_geometric.data import Data
import threading
from security import *
import random
import phe
from phe import EncodedNumber, paillier
#
from gen_local_emb import *
##嵌入模型
import sys

##
this_name = 'SN2'
##邻居表

def get_neighbor(neighbor_config_path = './neighbor.config'):
    with open(neighbor_config_path, encoding='utf-8') as f:
        res = []
        dic = {}
        neighbors = f.readlines()
        for n in neighbors:
            if n == '':break
            items = n.replace('\n','').split(',')
            items[1] = int(items[1])
            items = tuple(items)
            res.append(items)
            dic[items[2]] = (items[0], items[1])
    return res,dic
#加解密处理的是 tensor 向量 (1xn 或者 nx1)
def HE_encrypt(embed, pub_key):
    #处理精度
    x = embed.view(-1).tolist()
    
    return [pub_key.encrypt(e) for e in x]
def HE_decrypt(ciphertexts, pri_key):
    x = [pri_key.decrypt(c) for c in ciphertexts]
    return torch.DoubleTensor(x).view(-1,1)
gloab_table,gloab_table_dict = get_neighbor()
## 定义一个网络 作为模拟真实数据
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
# 节点的特征,m * n 的矩阵 可以认为是m个节点，在连续n天的感染情况                           
x = torch.tensor([[-1,2], [0,1], [1,-1]], dtype=torch.float)
x = x.T
class task(object):
    task_id : str
    edge : torch.Tensor
    x : torch.Tensor
    history_embed : dict #{迭代_id: 整图嵌入}

    def __init__(self,task_id, edge, x_) -> None:
        self.task_id = task_id
        self.edge = edge
        self.x = x_
    
task_item_tmp = {
    'task_id' : str,
    'task_item' : int,
    'task_content' : str,#train_Int, get_Global
    'train_times' : int
}

class SG(object):
    # task_id :  [taks_list]
    task_pool = {}
    # task_id : int
    task_iter = {}
    # task_batch  单道批处理
    task_batch = {}
    # task__enemb 任务对应的密文 二级结构 task_id -> name 
    task_enemb = {}
    # 同态加密的公钥,似乎没有隔离taskID
    task_HE_key = {}
    # @zerorpc.stream
    # 本节点的同态密钥
    task_self_pub = {}
    task_self_pri = {}
    # HE嵌入求和的数量(记录已经求和的数量，等于邻居数的时候，可以获取)
    task_HE_add_num = {}
    #嵌入的shape
    task_emb_shape = {}
    def iniItask(self, task_id:bytes, lr = 0.01):
        #解密
        task_id = decrypt(task_id)
        if task_id in self.task_pool.keys():
            return
        print('initia...', task_id)
        # return
        #初始化同态加密公私密钥
        public_key, private_key = paillier.generate_paillier_keypair(n_length=256)
        self.task_self_pri[task_id] = private_key
        self.task_self_pub[task_id] = public_key
        spub_key = pickle.dumps(public_key)
        #运行标志
        the_task = task(task_id, edge_index, x)
        the_task.Integrated_model = nn.Sequential(nn.Linear(x.shape[1],1)) 
        #定义一个Integrated 以及训练模块
        the_task.loss_func = torch.nn.MSELoss()
        self.task_pool[task_id] = the_task
        the_task.GLembed = self.task_pool[task_id].Integrated_model(x) 
        self.task_pool[task_id].optimizer = torch.optim.SGD(self.task_pool[task_id].Integrated_model.parameters(),lr = lr)
        #当前迭代轮次
        self.task_iter[task_id] = 0
        #开启taskBatch 序
        self.task_batch[task_id] = {}
        #初始化 task密文记录
        self.task_enemb[task_id] = {}
        #暂存明文
        task_id_text = task_id
        #初始化邻居
        for ne in gloab_table:
            cstring = ('tcp://%s:%d'%(ne[0],ne[1]))
            c = zerorpc.Client()
            c.connect(cstring)
            task_id = encrypt_by_name(task_id_text,ne[2])
            c.iniItask(task_id,lr)
            c.set_HE_pub_key(task_id_text, this_name, spub_key)
            c.close()
        # self.process(task_id_text)
    def get_other_global_embed(self,task_id, it):
        task_id = decrypt(task_id)
        if self.task_iter[task_id]>= it or it in self.task_batch[task_id].keys():
            return
        while it-1 > self.task_iter[task_id] :
            pass 
        task_item = task_item_tmp
        task_item['task_id'] = task_id
        task_item['task_content'] = 'get_Global'
        task_item['task_item'] = it
        self.task_batch[task_id][it] = task_item
        
        for ne in gloab_table:
            cstring = ('tcp://%s:%d'%(ne[0],ne[1]))
            c = zerorpc.Client()
            c.connect(cstring)
            c.get_other_global_embed(encrypt_by_name(task_id,ne[2]), it)
            c.close()
    def get_the_local_embed(self,task_id,ne_name, it):
        task_id = decrypt(task_id)
        while self.task_iter[task_id] < it-1:
            pass
        ##返回的嵌入
        x = self.task_pool[task_id].Integrated_model(self.task_pool[task_id].x)
        x = encrypt_by_name(x,ne_name)
        return x
    def train_IntergratedModel(self,task_id, it, times = 10):
        task_id = decrypt(task_id)
        ##已经执行或已有任务
        
        if it in self.task_batch[task_id].keys() or it <= self.task_iter[task_id]:
            return 
        
        task_item = task_item_tmp
        task_item['task_id'] = task_id
        task_item['task_content'] = 'train_Int'
        task_item['task_item'] = it
        task_item['train_times'] = times
        self.task_batch[task_id][it] = task_item

        ##传播到邻居
        for ne in gloab_table:
            cstring = ('tcp://%s:%d'%(ne[0],ne[1]))
            c = zerorpc.Client()
            c.connect(cstring)
            c.train_IntergratedModel(encrypt_by_name(task_id,ne[2]), it, times)
            c.close()
    ##同态加法喂数
    def HE_add(self,task_id, o_name,en_emb, it):
        en_emb = pickle.loads(en_emb)
        if o_name not in self.task_enemb[task_id].keys():
            x = self.task_pool[task_id].Integrated_model(self.task_pool[task_id].x)
            #加密
            x = HE_encrypt(x, pub_key = self.task_HE_key[task_id][o_name])
            self.task_enemb[task_id][o_name] = x
        if o_name == this_name:
            return 
        self.task_enemb[task_id][o_name] = \
            np.add(self.task_enemb[task_id][o_name], en_emb)
            
    ##被调用 以 传输本身的嵌入密文
    def trans_cipher_embed(self, task_id, o_name, t_name,it):
        encrypt_key =  self.task_HE_key[task_id][o_name]
        #获得嵌入并 加密
        x = self.task_pool[task_id].Integrated_model(self.task_pool[task_id].x)
        cipher_emb = HE_encrypt(x, pub_key = encrypt_key)
        cipher_emb = pickle.dumps(cipher_emb)
        #如果是本身，自传入
        if t_name == this_name:
            self.HE_add(task_id, o_name,cipher_emb,it)
            return
        target = gloab_table_dict[t_name]
        #传入计算节点
        cstring = ('tcp://%s:%d'%(target[0],target[1]))
        c = zerorpc.Client()
        c.connect(cstring)
        c.HE_add(task_id, o_name,cipher_emb,it)
        c.close()
    ##获取同态加密求和后的密文
    def get_HE_cipher_sum(self, task_id, o_name):
        return pickle.dumps(self.task_enemb[task_id][o_name])
    ##初始化过程中被配置的同态密钥, o_name and 密钥被pickdump 后
    def set_HE_pub_key(self, task_id, o_name, spub_key):
        if task_id not in self.task_HE_key.keys():
            self.task_HE_key[task_id] = {}
        self.task_HE_key[task_id][o_name] = pickle.loads(spub_key)
    def train_GGNN(self,task_id):
        res = 0
        nei_x = []
        for ne in gloab_table:
            cstring = ('tcp://%s:%d'%(ne[0],ne[1]))
            c = zerorpc.Client()
            c.connect(cstring)
            res += c.get_the_local_embed(task_id)
            nei_x.append(c.get_the_local_embed(task_id))
        self.task_pool[task_id].GGNN_model(torch.tensor(nei_x))
        prediction = self.task_pool[task_id].GGNN(self.task_pool[task_id].x)
        loss = self.task_pool[task_id].loss_func(prediction,torch.tensor([self.task_pool[task_id].GLembed]).to(torch.float))
        self.task_pool[task_id].GGNNoptimizer.zero_grad()
        loss.backward()
        self.task_pool[task_id].GGNNoptimizer.step()
    def SG_service(self, task_id:str):
        while True:
            ##结束标记
            # try:
            if self.task_iter[task_id] == -1:
                break
            if self.task_iter[task_id]+1 in self.task_batch[task_id].keys():
                task_item = self.task_batch[task_id][self.task_iter[task_id]+1]
                if task_item['task_content'] == 'train_Int':
                    print('train_IntergratedModel...')
                    for i in range(task_item['train_times']):
                        prediction = self.task_pool[task_id].Integrated_model(self.task_pool[task_id].x)
                        loss = self.task_pool[task_id].loss_func(prediction.float(),self.task_pool[task_id].GLembed.float())
                        self.task_pool[task_id].optimizer.zero_grad()
                        loss.backward()
                        self.task_pool[task_id].optimizer.step()
                        GLembed = decrypt(self.get_the_local_embed(encrypt_self(task_id),this_name,self.task_iter[task_id]+1))
                        self.task_pool[task_id].GLembed = GLembed
                    
                
                if task_item['task_content'] == 'get_Global':
                    print('get_other_global_embed...')
                    res = self.task_pool[task_id].GLembed

                    HEip,HEport,HEname = gloab_table[0][0],gloab_table[0][1],gloab_table[0][2]
                    for ne in gloab_table:
                        cstring = ('tcp://%s:%d'%(ne[0],ne[1]))
                        c = zerorpc.Client()
                        c.connect(cstring)
                        c.trans_cipher_embed(task_id, this_name, HEname, task_item['task_item'])
                        c.close()
                    #从计算节点获取同态加法后的密文，默认选举第一个邻居为
                    cstring = ('tcp://%s:%d'%(HEip,HEport))
                    c = zerorpc.Client()
                    c.connect(cstring)
                    res = pickle.loads(c.get_HE_cipher_sum(task_id, this_name))
                    c.close()
                    #加上自身
                    res = np.add(res, HE_encrypt(self.task_pool[task_id].GLembed, self.task_self_pub[task_id]))
                    res = HE_decrypt(res,self.task_self_pri[task_id])
                    #取平均
                    res /= (len(gloab_table) +1)
                    self.task_pool[task_id].GLembed = res
                self.task_iter[task_id]+=1
            # except:
            #     break
    def process(self,task_id):
        #args是关键字参数，需要加上名字，写成args=(self,)
        print(task_id)
        th1 = threading.Thread(target=self.SG_service, args=(task_id,))
        th1.start()
    def terminal(self, task_id):
        task_id = decrypt(task_id)
        if self.task_iter[task_id] == -1:
            return
        print('terminal...')
        self.task_iter[task_id] = -1
        for ne in gloab_table:
            cstring = ('tcp://%s:%d'%(ne[0],ne[1]))
            c = zerorpc.Client()
            c.connect(cstring)
            c.terminal(encrypt_by_name(task_id,ne[2]))
            c.close()
    def back_local_embed(self, task_id):
        
        task_id = decrypt(task_id)
        dic = dict(self.task_pool[task_id].Integrated_model.state_dict())
        bias = dic['0.bias']
        weight = dic['0.weight']
        resdiual = (self.task_pool[task_id].GLembed-bias) @ torch.linalg.pinv(weight.T)
        return encrypt_self(resdiual + self.task_pool[task_id].x)
        
        # return pickle.dumps((self.task_pool[task_id].GLembed-bias) @ torch.linalg.pinv(weight.T) )
    ##new way of get local embed
    def get_nei_embed(self, args, task_id):
        HEip,HEport,HEname = gloab_table[0][0],gloab_table[0][1],gloab_table[0][2]
        for ne in gloab_table:
            cstring = ('tcp://%s:%d'%(ne[0],ne[1]))
            c = zerorpc.Client()
            c.connect(cstring)
            c.train_and_trans_embed(args, task_id, this_name, HEname)
            #执行
            c.close()
        return HEip,HEport,HEname
    def train_and_trans_embed1(self, args:bytes,task_id:str, o_name:str, t_name:str):
        args = pickle.loads(args)
        #获取嵌入 tensor,多维 
        emb_ = get_local_emb(args)
        emb_shape = emb_.shape
        ## 展开成一维，用于同态加密
        emb_ = emb_.view(-1)
        HE_emh = HE_encrypt(emb_, pub_key=self.task_HE_key[task_id][o_name])
        HE_emh = pickle.dumps(HE_emh)
        #传播
        if t_name == this_name:
            self.HE_add_embed(task_id, o_name,HE_emh,emb_shape)
        else:
            target = gloab_table_dict[t_name]
            #传入计算节点
            cstring = ('tcp://%s:%d'%(target[0],target[1]))
            c = zerorpc.Client()
            c.connect(cstring)
            c.HE_add_embed(task_id, o_name,HE_emh,emb_shape)
            c.close()
    def HE_add_embed(self,task_id, o_name,en_emb, embed_shape):
        en_emb = pickle.loads(en_emb)
        if task_id not in self.task_emb_shape.keys():
            self.task_emb_shape[task_id] = embed_shape
        if task_id not in self.task_HE_add_num.keys():
            self.task_HE_add_num[task_id] = {}
        if o_name not in self.task_HE_add_num[task_id].keys():
            self.task_HE_add_num[task_id][o_name] = 0
        if o_name not in self.task_enemb[task_id].keys():
            self.task_enemb[task_id][o_name] = en_emb
            self.task_HE_add_num[task_id][o_name] += 1
            return

        self.task_enemb[task_id][o_name] = \
            np.add(self.task_enemb[task_id][o_name], en_emb)
        self.task_HE_add_num[task_id][o_name] += 1
    def get_HE_cipher_sum(self, task_id, o_name,nums_of_nei):
        if task_id not in self.task_HE_add_num.keys():
            return -1,-1
        if self.task_HE_add_num[task_id][o_name] < nums_of_nei:
            return -1,-1
        return pickle.dumps(self.task_enemb[task_id][o_name]), self.task_emb_shape[task_id]
    def get_nei_embed_sum(self, task_id, c_name):
        c_addr = gloab_table_dict[c_name]
        cstring = ('tcp://%s:%d'%(c_addr[0],c_addr[1]))
        c = zerorpc.Client()
        c.connect(cstring)
        cipher,emb_shape = c.get_HE_cipher_sum(task_id, this_name, len(gloab_table))
        c.close()
        if cipher == -1:
            return pickle.dumps(-1)
        return pickle.dumps(HE_decrypt(pickle.loads(cipher),self.task_self_pri[task_id]).reshape(emb_shape))
    def train_and_trans_embed(self,args:bytes,task_id:str, o_name:str, t_name:str):
        #args是关键字参数，需要加上名字，写成args=(self,)
        th1 = threading.Thread(target=self.train_and_trans_embed1, args=(args,task_id,o_name,t_name))
        th1.start()
##注册报告自身
# from SN import get_SN_peer
# get_SN_peer()

config = configparser.ConfigParser()
config.read('./config.ini',encoding='utf-8')
s = zerorpc.Server(SG())

s.bind("tcp://0.0.0.0:%s"%config.get('SG', 'port'))
# s.bind("tcp://127.0.0.1:%s"%'20002')
s.run()

# # class SGexcute(object):
# #     def __init__(self, model, data_name) -> None:
# #         #当前版本:模型是远程的，数据是本地的
# #         self.model = model
#         self.data = data_name
        
#     def get_local_graph_data(data_name):
        