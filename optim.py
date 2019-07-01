import numpy as np


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    l_r = config.get("learning_rate", 1e-2)
    w -= l_r * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    l_r = config.get("learning_rate", 1e-2)
    momentum = config.get("momentum", 0.9)
    vec = config.get("vec", np.zeros_like(w))
    vec = momentum * vec - l_r * dw
    w = w + vec
    config["vec"] = vec
    return w, config


def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta", 0.9)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("v", np.zeros_like(w))
    l_r = config["learning_rate"]
    beta = config["beta"]
    epsilon = config["epsilon"]
    v = config["v"]
    v = beta * v + (1 - beta)*(dw**2)
    config["v"] = v
    w -= l_r * dw / (np.sqrt(v) + epsilon)
    return w, config


def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)
    m = config['m']
    v = config['v']
    beta1 = config['beta1']
    beta2 = config['beta2']
    learning_rate = config['learning_rate']
    epsilon = config['epsilon']
    t = config['t']
    t += 1
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw**2)
    m_bias = m / (1 - beta1**t)
    v_bias = v / (1 - beta2**t)
    w += - learning_rate * m_bias / (np.sqrt(v_bias) + epsilon)
    config['m'] = m
    config['v'] = v
    config['t'] = t
    return w, config
