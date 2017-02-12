# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time
import logging

#ロガー
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%Y/%m/%d %H:%M:%S')

# add formatter to ch
handler.setFormatter(formatter)

logger.addHandler(handler)


'''
トレーニング済みのモデルから推論を行う

'''




def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=False, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    for i in range(1,4):
        key_w = "W" + str(i)
        key_b = "b" + str(i)
        print(key_w + "=" + str(network[key_w].shape))
        print(network[key_w])
        print(key_b + "=" + str(network[key_b].shape))
        print(network[key_b])

    return network


def predict(network, x):
    '''
    推論を行う

    Parameters
    ----------
    network
    x

    Returns
    -------

    '''
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']




    a1 = np.dot(x, W1) + b1
    logger.debug(a1)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y





def format_ndarray(nda):
    '''
    ndarray内の数値をroundする
    Parameters
    ----------
    nda

    Returns
    -------

    '''
    def _format(x):
        # x2 = round(x, 3)
        x2 = '{:.3f}'.format(x)
        # print("x=" + str(x) + " x2=" + str(x2))
        return str(x2)

    vf = np.vectorize(_format)
    v1 = vf(nda)

    # ndarrayのstr表現の横幅の制限を広くする
    v1 = np.array_str(v1, max_line_width=100000)
    return v1
    # print("v1=" + str(v1))

def img_show(img):
    from PIL import Image
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def main():
    """
    メイン処理
    Returns
    -------

    """
    logger.info("処理開始")
    start = time.time()
    x, t = get_data()
    x, t = x[0:100] ,t[0:100]

    network = init_network()
    accuracy_cnt = 0
    ng_count = 0
    batch_size = 100

    # 0〜100 101〜200
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]   #例 x[300:400]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        # バッチ内での正解数
        _accuracy_cnt = np.sum(p == t[i:i + batch_size])
        accuracy_cnt += _accuracy_cnt
        ng_count += (batch_size - _accuracy_cnt)


    end = time.time()

    logger.info("テスト件数:{0}".format(len(x)))
    logger.info("正解数:" + str(accuracy_cnt))
    logger.info("不正解数:" + str(ng_count))
    logger.info("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    logger.info("処理時間{0:.3f}".format(end - start))



if __name__ == '__main__':
    main()



