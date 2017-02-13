# coding: utf-8
import sys, os
import numpy as np
import dataset.mnist as mnist
from PIL import Image
sys.path.append(os.pardir)


print(mnist.__hoge1)
mnist.__hoge2('aaa')

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = mnist.load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)
