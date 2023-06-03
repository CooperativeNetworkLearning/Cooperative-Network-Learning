import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    dataset = 'state360'
    window = 20
    horizon = 10
    path = os.path.join('..','pic','plot_data')
    param_ = '_w{}h{}_'.format(window, horizon)

    gt = pd.read_csv(os.path.join(path, dataset + param_+ '_gt.csv'), header=None).values
    centralized = pd.read_csv(os.path.join(path, dataset + param_ + '_centralized.csv'), header=None).values
    local = pd.read_csv(os.path.join(path, dataset + param_ + '_local.csv'), header=None).values
    integrated = pd.read_csv(os.path.join(path, dataset +  param_ + '_integrated.csv'), header=None).values
    x = list(range(gt.shape[0]))

    plt.plot(x, np.mean(gt, axis=1), label='GT')
    plt.plot(x, np.mean(local, axis=1), label='Local')
    plt.plot(x, np.mean(integrated, axis=1), label='Integrated')
    plt.plot(x, np.mean(centralized, axis=1), label='Centralized')
    plt.title(dataset)
    plt.legend()
    plt.show()