import numpy as np
import optim


class Fit:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.x_train = data["x_train"]
        self.y_train = data["y_train"]
        self.x_val = data["x_val"]
        self.y_val = data["y_val"]
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 2)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        if len(kwargs) > 0:
            raise ValueError('wrong arg')
        if not hasattr(optim, self.update_rule):
            raise ValueError('wrong optim_args')
        self.update_rule = getattr(optim, self.update_rule)
        self._reset()

    def train(self):
        """训练若干次数"""
        num_train = self.x_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        for t in range(num_iterations):
            self._step()
            # 是否打印出当前轮次及损失值
            if self.verbose and t % self.print_every == 0:
                print('(iteration %d / %d) loss: %f' % (t + 1, num_iterations, self.loss_history[-1]))
            # 新的epoch学习率衰减
            if (t+1) % iterations_per_epoch == 0:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
                # 计算正确率，并记录最优模型参数
                train_acc = self.check_accuracy(self.x_train, self.y_train, num_samples=4)
                val_acc = self.check_accuracy(self.x_val, self.y_val, num_samples=4)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(train_acc)
                if self.verbose:
                    print('(epoch %d/%d) train_acc: %f val_acc: %f' % (self.epoch, self.num_epochs, train_acc, val_acc))
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.param_config.items():
                        self.best_params[k] = v.copy()
        # 将最优的参数赋值给对应模型
        self.model.param_config = self.best_params

    def _step(self):
        """一次前向传播和反向传播并更新参数"""
        # 从训练集中随机挑选样本
        num_train = self.x_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        # 使用模型求得损失及各个w的导数
        loss, grads = self.model.loss(x_batch, y_batch)
        self.loss_history.append(loss)
        # 使用优化方法，更新所有w
        for p, w in self.model.param_config.items():
            dw = grads[p]
            config = self.optim_configs[p]
            # print(p,w.shape,dw.shape)
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.param_config[p] = next_w
            self.optim_configs[p] = next_config

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.optim_configs = {}
        for p in self.model.param_config:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def check_accuracy(self, x, y, num_samples=None, batch_size=2):
        n = x.shape[0]
        if num_samples and n > num_samples:
            mask = np.random.choice(n, num_samples)
            x = x[mask]
            y = y[mask]
        # 计算batchs的数量
        num_batchs = num_samples // batch_size
        y_pred = []
        for i in range(num_batchs):
            start = i * batch_size
            end = (i+1) * batch_size
            scores = self.model.loss(x[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        # 将预测值平铺
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc
