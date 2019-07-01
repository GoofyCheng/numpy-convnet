import os
from layers import *


class SimpleCovnNet:
    """普通3层卷积网络模型"""
    def __init__(self, input_dim=(3, 32, 32), filter_num=32, filter_size=7,
                 hidden_dim=100, out_class=10, weight_scale=1e-3, reg=0.0, dtype=np.float32, params_path="conv_params.txt"):
        self.param_config = {}
        self.conv_config = {"stride": 1, "padding": 3}
        self.pool_config = {"stride": 2, "pool_height": 2, "pool_width": 2}
        self.reg = reg
        self.dtype = dtype
        self.params_path = params_path
        c, h, w = input_dim
        # 计算卷积后矩阵大小
        c_h = (self.conv_config["padding"]*2 + h - filter_size) // self.conv_config["stride"] + 1
        c_w = (self.conv_config["padding"]*2 + w - filter_size) // self.conv_config["stride"] + 1
        if not os.path.exists(self.params_path):
            self.param_config["W1"] = weight_scale * np.random.randn(filter_num, c, filter_size, filter_size)
            self.param_config["B1"] = weight_scale * np.random.randn(filter_num)
            # 3x32x32卷积后为32x32x32，池化后为32x16x16
            self.param_config["W2"] = weight_scale * np.random.randn(filter_num * c_h * c_w // 4, hidden_dim)
            self.param_config["B2"] = weight_scale * np.random.randn(hidden_dim)
            self.param_config["W3"] = weight_scale * np.random.randn(hidden_dim, out_class)
            self.param_config["B3"] = weight_scale * np.random.randn(out_class)
        else:
            with open(self.params_path, "r") as f:
                s = f.read()
                if s != "":
                    self.param_config = eval(s)
        for k, v in self.param_config.items():
            self.param_config[k] = v.astype(self.dtype)

    def loss(self, x, y=None):
        w1, b1 = self.param_config["W1"], self.param_config["B1"]
        w2, b2 = self.param_config["W2"], self.param_config["B2"]
        w3, b3 = self.param_config["W3"], self.param_config["B3"]
        # 卷积、激活、池化
        conv_out, c_cache = conv_relu_pool_forward(x, w1, b1, self.conv_config, self.pool_config)
        # 前向，激活
        hidden_out, h_cache = affine_relu_forward(conv_out, w2, b2)
        # 前向
        scores, s_cache = affine_forward(hidden_out, w3, b3)
        if y is None:
            return scores
        # softmax、交叉熵、反向求得dscores
        s_loss, dy = softmaxloss(scores, y)
        # 反向求得dhidden
        dhidden, dw3, db3 = affine_backward(dy, s_cache)
        # 反向求得dconv
        dconv, dw2, db2 = affine_relu_backward(dhidden, h_cache)
        # 反向求得dx
        dx, dw1, db1 = conv_relu_pool_backward(dconv, c_cache)
        # regularization
        dw1 += self.reg*w1
        dw2 += self.reg*w2
        dw3 += self.reg*w3
        reg_loss = 0.5 * self.reg * sum(np.sum(w*w) for w in [w1, w2, w3])
        loss = s_loss + reg_loss
        grads = {
            "W1": dw1, "W2": dw2, "W3": dw3, "B1": db1, "B2": db2, "B3": db3,
        }
        return loss, grads

