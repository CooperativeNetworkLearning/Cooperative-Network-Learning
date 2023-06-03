import pickle
import argparse
import os
import subprocess
import multiprocessing
gpu_num = 8
n = 8
def worker(task):
    subprocess.Popen(["python", "exp.py","--seed",task[0],"--device",task[1]])


if __name__ == '__main__':
    # 原始列表

    

    data = [(str(s),str(s%gpu_num)) for s in range(2,10)]

    # 创建进程池
    pool = multiprocessing.Pool(processes=n)

    # 并行处理
    pool.map(worker, data)


    print('end')