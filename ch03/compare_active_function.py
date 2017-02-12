# coding: utf-8
import sys
import os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pylab as plt
import common.takemura as tk

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)
y3 = relu(x)

if __name__ == '__main__':
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylim(-0.1, 1.1)
    tk.showActivateWindow()
    # tk.plt_show_focus(plt)
