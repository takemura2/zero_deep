# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
import sys,os
sys.path.append(os.pardir)
import common.takemura as tk

def step_function(x):
    return np.array(x > 0, dtype=np.int)

if __name__ == '__main__':
    X = np.arange(-5.0, 5.0, 0.1)
    Y = step_function(X)
    plt.plot(X, Y)
    plt.ylim(-0.1, 1.1)  # 図で描画するy軸の範囲を指定
    tk.plt_show_focus(plt)
