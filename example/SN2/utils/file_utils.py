
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

def get_log_path(args, runTag=None):
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


