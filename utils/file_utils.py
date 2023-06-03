
import torch
import os
import utils.data_utils

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


def get_subInstitution(dataset, clientNum, institueion_idx):
    if institueion_idx is None:
        institution_name = dataset + '-global'
        localIndex = None
    else:
        institution_name = dataset + '-sub' + chr(ord('A') + institueion_idx)
        _, institution_list = utils.data_utils.gen_institution(dataset, clientNum)# 获取机构划分
        localIndex = institution_list[institueion_idx]

    return institution_name, localIndex


def get_share_file_path(dir_name):
    return os.path.join('..', 'share_emb', dir_name)

def get_local_file_path(dir_name):
    # 运行过程中本地文件存储路径
    # 运行过程中会生成xx_log.yaml, 模型文件
    return os.path.join('..', 'src', 'local_log', dir_name)

def get_log_path(args, runTag=None):
    """
    日志文件存储路径
    """
    if runTag is None:
        runTag = args.runTag

    return os.path.join(get_local_file_path(args.dir_name), args.institution_name + '_runTag' + str(runTag) + "_log.yaml")

def get_temp_model_dir(args):
    temp_model_dir = os.path.join(get_local_file_path(args.dir_name), 'model',
                                  '{}_runTag{}_tempModel.pt'.format(args.institution_name, args.runTag))
    return temp_model_dir

def get_best_model_dir(args, runTag=None):
    if runTag is None:
        runTag = args.runTag

    best_model_dir = os.path.join(get_local_file_path(args.dir_name), 'model',
                                  '{}_runTag{}_bestModel.pt'.format(args.institution_name, runTag))
    return best_model_dir




def get_shareEmb_path(args, runTag=None):
    """
    共享嵌入文件存储路径
    """
    if runTag is None:
        runTag = args.runTag

    return os.path.join(get_share_file_path(args.dir_name), args.institution_name + '_runTag' + str(runTag) + '_emb.pt')



def get_shareEmb(args, n_sample):
    """
    获取共享池中的嵌入
    """
    share_emb = torch.zeros((n_sample, 1, args.hidR))
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