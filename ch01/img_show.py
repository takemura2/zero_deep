# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import common.takemura as tk
import sys,os
sys.path.append(os.pardir)
img = imread('../dataset/lena.png')

plt.imshow(img)
tk.showActivateWindow()
