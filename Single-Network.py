# -*- coding:utf-8 -*-
import logging
import random
import mxnet as mx  # 导入 MXNet 库
import numpy as np  # 导入 NumPy 库，这是 Python 常用的科学计算库

logging.getLogger().setLevel(logging.DEBUG)  # 打开调试信息的显示

n_sample = 1000  # 训练用的数据点个数
batch_size = 1  # 批大小
learning_rate = 0.1  # 学习速率
n_epoch = 1  # 训练 epoch 数

# 每个数据点是在 (0,1) 之间的 2 个随机数
train_in = [[random.uniform(0, 1) for c in range(2)] for n in range(n_sample)]

train_out = [0 for n in range(n_sample)]  # 期望输出，先初始化为 0

for i in range(n_sample):
    # 每个数据点的期望输出是 2 个输入数中的大者
    train_out[i] = max(train_in[i][0], train_in[i][1])

# 数组迭代器
train_iter = mx.io.NDArrayIter(data=np.array(train_in),
                               label={'reg_label': np.array(train_out)},
                               batch_size=batch_size,
                               shuffle=True #随机打乱数据
                               )

src = mx.sym.Variable('data')

fc = mx.sym.FullyConnected(data=src, num_hidden=1, name='fc')

net = mx.sym.LinearRegressionOutput(data=fc, name='reg')

module = mx.mod.Module(symbol=net, label_names=(['reg_label']))


def epoch_callback(epoch, symbol, arg_params, aux_params):
    for k in arg_params:  # 对于所有参数…
        print(k)  # 输出参数名
        print(arg_params[k].asnumpy())  # 参数值，转为 NumPy 数组，输出更美观


module.fit(
    train_iter,  # 训练数据的迭代器
    eval_data=None,  # 在此只训练，不使用测试数据
    eval_metric=mx.metric.create('mse'),  # 输出 MSE 损失信息
    optimizer='sgd',  # 梯度下降算法为 SGD
    optimizer_params={'learning_rate': learning_rate},  # 设置学习速率
    num_epoch=n_epoch,  # 训练 epoch 数
    batch_end_callback=mx.callback.Speedometer(batch_size, 100),    # 每经过 100 个 batch 输出训练速度
    epoch_end_callback=epoch_callback,  # 完成每个 epoch 后调用 epoch_callback
)
