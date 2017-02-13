# coding: utf-8
import sys
import os
import numpy as np
import matplotlib.pylab as plt
import common.takemura as tk
sys.path.append(os.pardir)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def step_function(x):

    # 下記のように書いても良い
    # b = x > 0
    # return b.astype(np.int)

    return np.array(x > 0, dtype=np.int)


def relu(x):
    return np.maximum(0, x)


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = step_function(x)
    y3 = relu(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylim(-0.1, 1.1)
    tk.showActivateWindow()
    # tk.plt_show_focus(plt)
