import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as DataLoaderG
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import *
dataset = TUDataset(root='d:/data/TUDataset', name='MUTAG')

torch.manual_seed(12345)
dataset = dataset.shuffle()
dataset_raw = dataset
train_dataset = dataset[:20]
test_dataset = dataset[100:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoaderG(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoaderG(test_dataset, batch_size=1, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    def embed(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
#预训练阶段
def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.

# max_acc1=0
# for epoch in range(1, 101):
#     train()
    # train_acc = test(train_loader)
    # test_acc = test(test_loader)
    # if test_acc>max_acc1:
    #     max_acc1 = test_acc
    # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

model.eval()

##注册报告自身
from SN import get_SN_peer
import zerorpc
import pickle
import uuid
from security import *

total_embed = []
total_y = []

gloab_table = [
    ('127.0.0.1',20001),
    # ('127.0.0.1',20002)
]
##初始化
task_id = str(uuid.uuid4())
print(task_id)
for t in gloab_table:
    cstring = ('tcp://%s:%d'%(t[0],t[1]))
    pu = loadPublicKey('./public.pem')

    text = pickle.dumps(task_id)
    task_id = encrypt(text, pu)

    c = zerorpc.Client()
    c.connect(cstring)
    c.iniItask(task_id)
    c.close()
for i in range(1,5):
    it = 2*i
    for t in gloab_table:
        cstring = ('tcp://%s:%d'%(t[0],t[1]))
        c = zerorpc.Client()
        c.connect(cstring)
        c.get_other_global_embed(task_id,it-1)
        c.close()
    for t in gloab_table:
        cstring = ('tcp://%s:%d'%(t[0],t[1]))
        c = zerorpc.Client()
        c.connect(cstring)
        c.train_IntergratedModel(task_id, it)
        c.close()
c = zerorpc.Client()
c.connect(cstring)
x = c.back_local_embed(task_id)
print(pickle.loads(x))
c.terminal(task_id)

# for sn in get_SN_peer():
#     print('getting ',sn)
#     cstring = ('tcp://%s:%d'%(sn['ip'],sn['port']))
#     c = zerorpc.Client()
#     c.connect(cstring)
#     x,y = c.pullGraphEmbed('rawGNN2', 'TU')
#     x = pickle.loads(x)
#     y = pickle.loads(y) 
#     total_embed.append(x)
#     total_y.append(y)
# total_embed = torch.concat(total_embed, 0)
# total_y = torch.concat(total_y, 0)

## 再训练最后的分类器
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# dataset = TensorDataset(total_embed, total_y)

# train_loader = DataLoader(dataset = dataset, batch_size=16)

# class last_net(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(last_net, self).__init__()
#         torch.manual_seed(12345)
#         self.lin = Linear(hidden_channels, 2)

#     def forward(self, x):
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#         return x
# net = last_net(hidden_channels = 64)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

# def train():
#     net.train()
#     for data in train_loader:  # Iterate in batches over the training dataset.
#         x, y = data
#         out = net(x)  # Perform a single forward pass.
#         loss = criterion(out, y)  # Compute the loss.
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.
# def test(loader):
#     net.eval()
#     correct = 0
#     for data in loader:  # Iterate in batches over the training/test dataset.
#         out = net(model.embed(data.x, data.edge_index, data.batch))
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#     return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.
# max_acc2 = 0
# for epoch in range(1, 101):
#     train()
#     test_acc = test(test_loader)
#     if test_acc>max_acc2:max_acc2=test_acc
#     print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')



# print(' acc2: %f'%(max_acc2))


# #再测试基准准确率
# train_dataset = dataset_raw[:1]
# bench_loader = DataLoaderG(train_dataset, batch_size=len(train_dataset), shuffle=True)
# for data in bench_loader:
#     dataset_x = model.embed(data.x, data.edge_index, data.batch)
#     dataset_y = data.y
# bench_model = last_net(hidden_channels = 64)
# optimizer = torch.optim.Adam(bench_model.parameters(), lr=0.01)
# def bench_train():
#     bench_model.train()
#     for data in train_loader:  # Iterate in batches over the training dataset.
#         x, y = data
#         out = bench_model(x)  # Perform a single forward pass.
#         loss = criterion(out, y)  # Compute the loss.
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.
# def bench_test(loader):
#     bench_model.eval()
#     correct = 0
#     for data in loader:  # Iterate in batches over the training/test dataset.
#         out = bench_model(model.embed(data.x, data.edge_index, data.batch))
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#     return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.
# max_acc1 = 0
# for epoch in range(1, 101):
#     bench_train()
#     test_acc = bench_test(test_loader)
#     if test_acc>max_acc1:max_acc1=test_acc
#     print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')
# print(max_acc1, max_acc2)
# print([test_dataset[i].y for i in range(len(test_dataset))])