
import torch
import os




def get_shareEmb(args, n_sample):
    """
    获取共享池中的嵌入
    """
    share_emb = torch.zeros((n_sample, 1, args.hidR))
    t = 0
    for file_name in os.listdir(args.trustGNNpath):
        # 读取相同runTag的邻居嵌入
        if (str(args.runTag - 1) not in file_name) or ('_emb.pt' not in file_name):
            continue
        # 不读取自身生成的嵌入
        if file_name != args.institution_name + '_runTag' + str(args.runTag - 1) + '_emb.pt':
            share_emb = share_emb + torch.load(os.path.join(args.trustGNNpath, file_name))
            t += 1

    return share_emb/t

def get_log_path_node(args, runTag=None):
    """
    日志文件存储路径
    """
    if runTag is None:
        runTag = args.runTag

    return os.path.join(args.local_log_path, args.institution_name + '_runTag' + str(runTag) + "_log.yaml")


def get_emb_path(args, runTag=None):
    """
    嵌入文件存储路径
    """
    if runTag is None:
        runTag = args.runTag

    return os.path.join(args.trustGNNpath, args.institution_name + '_runTag' + str(runTag) + '_emb.pt')


import os
import torch

ROOT_DIR = '.'

def gen_CNL_dir(dir_name):
    """ 生成程序运行过程中需要的文件夹 """

    # 共享文件路径
    if not os.path.exists( get_share_file_path(dir_name)):
        os.makedirs( get_share_file_path(dir_name))

    # 运行过程中本地文件存储路径
    if not os.path.exists( get_local_file_path(dir_name)):
        os.makedirs( get_local_file_path(dir_name))

    # 存储模型的路径
    if not os.path.exists(os.path.join(get_local_file_path(dir_name), 'model')):
        os.makedirs(os.path.join(get_local_file_path(dir_name), 'model'))


def get_share_file_path(dir_name):
    return os.path.join(ROOT_DIR, 'share_emb', dir_name)

def get_local_file_path(dir_name):
    # 运行过程中本地文件存储路径
    # 运行过程中会生成xx_log.yaml, 模型文件
    return os.path.join(ROOT_DIR, 'local_log', dir_name)

def get_log_path(args, runTag=None):
    """
    日志文件存储路径
    """
    if runTag is None:
        runTag = args.runTag

    return os.path.join(get_local_file_path(args.dir_name), args.institution_name + '_runTag' + str(runTag) + "_log.yaml")

def get_temp_model_dir(args):
    temp_model_dir = os.path.join(get_local_file_path(args.dir_name), 'model',
                                  f'{args.institution_name}_runTag{args.runTag}_tempModel.pt')
    return temp_model_dir

def get_best_model_dir(args, runTag=None):
    if runTag is None:
        runTag = args.runTag

    best_model_dir = os.path.join(get_local_file_path(args.dir_name), 'model',
                                  f'{args.institution_name}_runTag{runTag}_bestModel.pt')
    return best_model_dir

def get_shareEmb_path(args, runTag=None):
    """
    共享嵌入文件存储路径
    """
    if runTag is None:
        runTag = args.runTag

    return os.path.join(get_share_file_path(args.dir_name), args.institution_name + '_runTag' + str(runTag) + '_emb.pt')

def get_shareEmb(args):
    """
    获取共享池中的嵌入
    """
    self_emb = torch.load(get_shareEmb_path(args,runTag=-1))
    share_emb = torch.zeros(self_emb.size(), device=self_emb.device)


    t = 0
    share_path = get_share_file_path(args.dir_name)
    for file_name in os.listdir(share_path):
        # 读取相同runTag的邻居嵌入
        if (str(args.runTag - 1) not in file_name) or ('_emb.pt' not in file_name):
            continue
        # 不读取自身生成的嵌入
        if file_name != args.institution_name + '_runTag' + str(args.runTag - 1) + '_emb.pt':
            share_emb = share_emb + torch.load(os.path.join(share_path, file_name))
            t += 1

    return share_emb/t