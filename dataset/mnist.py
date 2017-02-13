# coding: utf-8
import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'

# gunzip前
# key_file = {
#     'train_img':'train-images-idx3-ubyte.gz',
#     'train_label':'train-labels-idx1-ubyte.gz',
#     'test_img':'t10k-images-idx3-ubyte.gz',
#     'test_label':'t10k-labels-idx1-ubyte.gz'
# }

# gunzip後
key_file = {
    'train_img':'train-images-idx3-ubyte',
    'train_label':'train-labels-idx1-ubyte',
    'test_img':'t10k-images-idx3-ubyte',
    'test_label':'t10k-labels-idx1-ubyte'
}


dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 28 * 28

__hoge1 = 'abc'

def __hoge2(str):
    print("mnist.__hoge2:" + str)



def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            # np.uint8 8ビット符号無し整数
            # offset=16 16byte目から画像ファイルがある
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み
    
    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label : 
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか 
    
    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        # 1byte(256)を0〜1の間に正規化
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def _load_img2(file_name):
    '''
    gunzip後の生データ読み込み
    先頭の16byteがヘッダ部分で残りが画像データ
    1つの画像は28x28 pixel
    1pixel 8bit = 1byte (256)

    トレーニングデータサイズ
    ヘッダ16byte + 28byte x 28 byte x 60,000枚 = 47040000byte

    Parameters
    ----------
    file_name 読み込む画像ファイルのパス

    Returns shape(画像数,1画像のpixel数)のndarray
    -------

    '''
    file_path = dataset_dir + "/" + file_name

    print("Converting img" + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
        # np.uint8 8ビット符号無し整数
        # offset=16 16byte目から画像ファイルがある
        data = np.frombuffer(f.read(), np.uint8, offset=16) # type: np.ndarray
        # np.uint8 (1byteの１次元配列)
        print('shape=' + str(data.shape))

        print('reshape前のndarray=' + str(data))

    # 1次元配列から 28*28=784byteづつ１画像単位に切る（行に-1を指定すると自動的に分解される）
    data = data.reshape(-1, img_size)
    print("reshape後のndarray.shape=" + str(data.shape))
    print("reshape後のdata=" + str(data))
    print("Done")

    return data


def _load_label2(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting label" + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    print("shape=" + str(labels.shape))
    print("labels=" + str(labels))
    print("Done")

    return labels


#確認
# train_img   = _load_img2(key_file['train_img'])
# train_label =_load_label2(key_file['train_label'])

if __name__ == '__main__':
    pass
    # init_mnist()
