# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import common.takemura as tk
img = imread('../dataset/lena.png')

plt.imshow(img)
tk.showActivateWindow()
