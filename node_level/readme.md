# CNL框架处理节点级任务（节点分类）

## 文件目录说明
* data: 数据文件
* local_log: CNL本地日志文件，会通过比较val_loss挑选最优模型
* share_emb： 共享的嵌入


## 代码说明
* file_utils: 处理文件的函数
* data_utils: 处理数据的函数
* models.py: GNN模型文件
* GNN_local_batch.py: local模型，institueion_idx=None则不进行机构划分，即是centralized的模型
* GNN_integrated_batch.py: integrated 模型，runTag=0,1,...
* nei_cluster.ipynb: 通过谱划分的方式进行机构划分
* print_rel.ipynb: 集中展示结果（读取local_log中的记录）