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
logger.setLevel(logging.INFO)

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

    network = init_network()
    accuracy_cnt = 0
    ng_count = 0

    for i in range(len(x)):
    # for i in range(1):
        y = predict(network, x[i])

        # 画像を表示したい場合
        # if i == 8:
        #     img_show(x[i].reshape((28, 28)))

        p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
        if p == t[i]:
            accuracy_cnt += 1
        else:
            ng_count += 1
            logger.debug("ハズレ! i=" + str(i) + " t=" + str(t[i]) + " p=" + str(p) + " y=" + str(format_ndarray(y)))



    end = time.time()

    logger.info("テスト件数:{0}".format(len(x)))
    logger.info("正解数:" + str(accuracy_cnt))
    logger.info("不正解数:" + str(ng_count))
    logger.info("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    logger.info("処理時間{0:.3f}".format(end - start))

if __name__ == '__main__':
    main()