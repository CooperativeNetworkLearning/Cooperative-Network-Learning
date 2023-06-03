1. 启动三个终端，分别进入sn1,sn2,sn3目录(模拟三个机构)

2. 对每个终端，分别执行
```bash
python.exe ./SG.py
##应该注意python运行环境
```
启动后会自动根据配置文件开启本地*20001~20003*

3. 根据嵌入的获取更新方法，融合了一鸣的代码，为了方便可动态调节，采用args的方式作为参数输入。
```python
##配置任务参数
import os, itertools, random, argparse, time, datetime,torch
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='region785', help="Dataset string")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)")
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)")
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
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
args = ap.parse_args([])

args.runTag = -1  # -1说明是初始生成轮,该文件中必须为-1
args.global_nei_num = 0  # 邻居数量，该文件中必须为0
args.hidR = args.k * 4 * args.hidP + args.k #不能修改

```

- 执行命令，获取邻居嵌入

``` python
import zerorpc
import uuid
from SN1.security import *
##初始化
task_id = str(uuid.uuid4())
raw_task_id = task_id
print(task_id)
pu = loadPublicKey('./SN1/public.pem')
text = pickle.dumps(task_id)
task_id = encrypt(text, pu)

cstring = ('tcp://%s:%d'%('127.0.0.1',20001))
c = zerorpc.Client(heartbeat=None)
c.connect(cstring)
c.iniItask(task_id)
_,_,d = c.get_nei_embed(pickle.dumps(args), raw_task_id)
while True:
    ##等待计算完毕，每10秒检测1次
    a = pickle.loads(c.get_nei_embed_sum(raw_task_id, d))
    if not isinstance(a,int):
        res = a
        break
    time.sleep(10)
#邻居嵌入之和
res
```