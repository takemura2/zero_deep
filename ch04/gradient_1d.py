# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

from common.takemura import *


def numerical_diff_normal(f, x):
    '''数値微分　前方差分'''
    h = 1e-4
    return (f(x + h) - f(x) / h)

def numerical_diff(f, x):
    '''数値微分　中央差分'''
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 3*x**2 + 3*x


def tangent_line(f, x):
    '''関数fの接線関数を返す'''

    d = numerical_diff(f, x)  # 傾き
    print(d)
    b = f(x) - d*x  # f(x) = d*x + b -> b = f(x) -d*x
    # print(b)

    # f(x) = ax + b
    return lambda t: d*t + b
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")


# tf = tangent_line(function_1, 5)
# y2 = tf(x)

x2 = np.arange(5, 15, 5)
for x3 in x2:
    tf3 = tangent_line(function_1, x3)
    y3 = tf3(x)
    plt.plot(x, y3)

plt.plot(x, y)
# plt.plot(x, y2)
plt_show_focus(plt)
plt.show()
