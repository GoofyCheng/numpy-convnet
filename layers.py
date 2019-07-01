import numpy as np


def Relu(x):
    return np.maximum(x, 0)


def affine_forward(x, w, b):
    y = None
    n = x.shape[0]
    x_row = x.reshape(n, -1)
    y = np.dot(x_row, w) + b
    return y, (x, w, b)


def affine_backward(dout, cache):
    x, w, b = cache
    n = x.shape[0]
    x_row = x.reshape(n, -1)
    # 将dx修改为x的形状
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
    # 这里需要将x平整后才能dot
    dw = np.dot(x_row.T, dout)
    # db需要保持与dy相同的维度
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = None
    out = Relu(x)
    return out, x


def relu_backward(dout, cache):
    x = cache
    dx = dout
    # dx = (x > 0) * dout
    dx[x <= 0] = 0
    return dx


def affine_relu_forward(x, w, b):
    """一层神经网络（包括relu激活）"""
    y, cache = affine_forward(x, w, b)
    out, r_cache = relu_forward(y)
    return out, (cache, r_cache)


def affine_relu_backward(dout, cache):
    """向后传播（包括relu激活）"""
    f_cache, r_cache = cache
    dy = relu_backward(dout, r_cache)
    dx, dw, db = affine_backward(dy, f_cache)
    return dx, dw, db


def conv_forward_naive(x, w, b, conv_config):
    """普通卷积"""
    n, c, height, width = x.shape
    f_n, f_c, f_h, f_w = w.shape
    stride, padding = conv_config["stride"], conv_config["padding"]
    # 求对应卷积后的形状
    o_h = 1 + (height + 2*padding - f_h) // stride
    o_w = 1 + (width + 2*padding - f_w) // stride
    # 将输入进行padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    out = np.zeros((n, f_n, o_h, o_w))
    for i in range(n):
        for j in range(f_n):
            for k in range(o_h):
                for l in range(o_w):
                    # 对应每个filter将图片每个channel卷积相加并加上偏置
                    out[i, j, k, l] = np.sum(x_pad[i, :, stride*k:stride*k + f_h, stride*l:stride*l + f_w]*w[j]) + b[j]
    cache = (x, w, b, conv_config)
    return out, cache


def conv_backward_naive(dout, cache):
    """普通卷积的反向传播"""
    x, w, b, conv_config = cache
    stride, padding = conv_config["stride"], conv_config["padding"]
    n, c, height, width = x.shape
    f_n, c, f_h, f_w = w.shape
    o_h = 1 + (height + 2*padding - f_h) // stride
    o_w = 1 + (width + 2*padding - f_w) // stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    dx_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    db = np.zeros_like(b)
    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    for i in range(n):
        for j in range(f_n):
            for k in range(o_h):
                for l in range(o_w):
                    window = x_pad[i, :, k*stride:k*stride + f_h, l*stride:l*stride + f_w]
                    db[j] += dout[i, j, k, l]
                    dw[j] += window * dout[i, j, k, l]
                    dx_pad[i, :, k*stride:k*stride + f_h, l*stride:l*stride + f_w] += dout[i, j, k, l] * w[j]
    dx = dx_pad[:, :, padding:height + padding, padding:width + padding]
    return dx, dw, db


def max_pool_forward_naive(x, pool_config):
    """最大值池化前向传播"""
    p_h, p_w, stride = pool_config["pool_height"], pool_config["pool_width"], pool_config["stride"]
    n, c, h, w = x.shape
    # 求取池化后形状，勿忘+1
    o_h = (h - p_h) // stride + 1
    o_w = (w - p_w) // stride + 1
    out = np.zeros((n, c, o_h, o_w))
    for i in range(n):
        for j in range(c):
            for k in range(o_h):
                for l in range(o_w):
                    out[i, j, k, l] = np.max(x[i, j, stride*k:stride*k + p_h, stride*l:stride*l + p_w])
    cache = (x, pool_config)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """最大值池化的反向传播"""
    x, pool_config = cache
    p_h, p_w, stride = pool_config["pool_height"], pool_config["pool_width"], pool_config["stride"]
    n, c, h, w = x.shape
    o_h = (h - p_h) // stride
    o_w = (w - p_w) // stride
    dx = np.zeros_like(x)
    for i in range(n):
        for j in range(c):
            for k in range(o_h):
                for l in range(o_w):
                    window = x[i, j, stride*k:stride*k + p_h, stride*l:stride*l + p_w]
                    m = np.max(window)
                    dx[i, j, stride*k:stride*k + p_h, stride*l:stride*l + p_w] = (window == m)*dout[i, j, k, l]
    return dx


def conv_relu_pool_forward(x, w, b, conv_config, pool_config):
    """卷积-relu激活-最大值池化的前向传播"""
    # 步骤：卷积， relu激活, 池化
    conv_out, c_cache = conv_forward_naive(x, w, b, conv_config)
    relu_out, r_cache = relu_forward(conv_out)
    pool_out, p_cache = max_pool_forward_naive(relu_out, pool_config)
    return pool_out, (c_cache, r_cache, p_cache)


def conv_relu_pool_backward(dout, cache):
    """池化-relu激活-最大值池化的反向传播"""
    c_cache, r_cache, p_cache = cache
    dpool_out = max_pool_backward_naive(dout, p_cache)
    drelu_out = relu_backward(dpool_out, r_cache)
    dx, dw, db = conv_backward_naive(drelu_out, c_cache)
    return dx, dw, db


def softmax(x):
    """输出的softmax"""
    # 求最大值的坐标为1，需要保持矩阵的形状
    x -= np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    exp /= np.sum(exp, axis=1, keepdims=True)
    return exp


def softmaxloss(x, y):
    """softmax、交叉熵定义损失，反向求导"""
    probs = softmax(x)
    N = probs.shape[0]
    loss = -np.sum(np.log(probs[range(N), y])) / N
    dx = probs.copy()
    # 交叉熵求导为：p - y
    dx[range(N), y] -= 1
    # 除以N个样本
    dx /= N
    return loss, dx
