# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import common.takemura as tk


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


if __name__ == '__main__':
    # X = np.arange(-100.0, 100.0, 0.1)
    X = np.arange(-10.0, 10.0, 0.1)
    Y = sigmoid(X)
    plt.plot(X, Y)
    plt.ylim(-0.1, 1.1)

    tk.plt_show_focus(plt)


