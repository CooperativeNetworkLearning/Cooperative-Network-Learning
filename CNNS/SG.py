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
## read configure info
config = configparser.ConfigParser()
config.read('./config.ini',encoding='utf-8')
##
this_name = config.get('SG', 'name')
institueion_idx = int(this_name[2:]) - 1 
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

def HE_encrypt(embed, pub_key):

    x = embed.view(-1).tolist()
    
    return [pub_key.encrypt(e) for e in x]
def HE_decrypt(ciphertexts, pri_key):
    x = [pri_key.decrypt(c) for c in ciphertexts]
    return torch.DoubleTensor(x).view(-1,1)
gloab_table,gloab_table_dict = get_neighbor()

class task(object):
    task_id : str
    edge : torch.Tensor
    x : torch.Tensor
    history_embed : dict

    
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

    task_batch = {}
    # task__enemb 
    task_enemb = {}
    # 
    task_HE_key = {}
    # @zerorpc.stream
    # 本节点的同态密钥
    task_self_pub = {}
    task_self_pri = {}
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
        the_task = task()
        self.task_pool[task_id] = the_task
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

    def get_HE_cipher_sum(self, task_id, o_name):
        return pickle.dumps(self.task_enemb[task_id][o_name])
    ##初始化过程中被配置的同态密钥, o_name and 密钥被pickdump 后
    def set_HE_pub_key(self, task_id, o_name, spub_key):
        if task_id not in self.task_HE_key.keys():
            self.task_HE_key[task_id] = {}
        self.task_HE_key[task_id][o_name] = pickle.loads(spub_key)

    def process(self,task_id):

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
        args.institueion_idx = institueion_idx
        emb_ = get_local_emb_switch(args)
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
        return pickle.dumps(HE_decrypt(pickle.loads(cipher),self.task_self_pri[task_id]).reshape(emb_shape)/gloab_table.__len__())
    def train_and_trans_embed(self,args:bytes,task_id:str, o_name:str, t_name:str):
        th1 = threading.Thread(target=self.train_and_trans_embed1, args=(args,task_id,o_name,t_name))
        th1.start()

s = zerorpc.Server(SG())

s.bind("tcp://0.0.0.0:%s"%config.get('SG', 'port'))
s.run()

        