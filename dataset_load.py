import pickle
import os
import numpy as np


def load_cifar_batch(file):
    """加载cifar的某个batch"""
    with open(file, "rb") as f:
        data = pickle.load(f, encoding="latin1")
        x = data['data']
        y = data['labels']
        xt = x.reshape(10000, 3, 32, 32).transpose(0, 3, 2, 1).astype('float')
        yt = np.array(y)
        return xt, yt


def load_cifar(root):
    """加载cifar数据集所有batch"""
    xt = []
    yt = []
    for i in range(1, 6):
        cifar_dir = os.path.join(root, "data_batch_%d" % i)
        x, y = load_cifar_batch(cifar_dir)
        xt.append(x)
        yt.append(y)
    xtr = np.concatenate(xt)
    ytr = np.concatenate(yt)
    xte, yte = load_cifar_batch(os.path.join(root, "test_batch"))
    return xtr, ytr, xte, yte


def get_cifar_data(num_training=5000, num_validation=500, num_test=500):
    """加载cifar所有数据，并选择部分作为训练数据、验证数据、测试数据"""
    cifar_d = "D://cifar-10-batches-py//"
    xtr, ytr, xte, yte = load_cifar(cifar_d)
    mask = range(num_training)
    x_train = xtr[mask]
    y_train = ytr[mask]
    mask = range(num_training, num_training + num_validation)
    x_val = xtr[mask]
    y_val = ytr[mask]
    mask = range(num_test)
    x_test = xte[mask]
    y_test = yte[mask]
    # normalizer
    xtr_mean = np.mean(xtr, axis=0)
    x_train -= xtr_mean
    x_val -= xtr_mean
    x_test -= xtr_mean
    # transpose
    x_train = x_train.transpose(0, 3, 1, 2)
    x_val = x_val.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }
