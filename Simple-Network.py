# -*- coding:utf-8 -*-
import logging
import random
import mxnet as mx
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)

n_simple = 1000
batch_size = 1
learning_rate = 0.1
n_epoch = 1

train_in = [[random.uniform(0, 1) for c in range(2)] for n in range(n_simple)]
train_out = [0 for n in range(n_simple)]

for i in range(n_simple):
    train_out[i] = max(train_in[i][0], train_in[i][1])

train_iter = mx.io.NDArrayIter(
    data=np.array(train_in),
    label={'reg_label': np.array(train_out)},
    batch_size=batch_size,
    shuffle=True
)

src = mx.sym.Variable('data')
fct1 = mx.sym.FullyConnected(name='fct1', data=src, num_hidden=10)
act1 = mx.sym.Activation(name='act1', data=fct1, act_type='relu')
fct2 = mx.sym.FullyConnected(name='fct2', data=act1, num_hidden=10)
act2 = mx.sym.Activation(name='act2', data=fct2, act_type='relu')
fct3 = mx.sym.FullyConnected(name='fct3', data=act2, num_hidden=1)
net = mx.sym.LinearRegressionOutput(name='reg', data=fct3)
module = mx.mod.Module(symbol=net, label_names=(['reg_label']))

module.fit(
    train_iter,
    eval_data=None,
    eval_metric=mx.metric.create('mse'),
    initializer=mx.initializer.Uniform(0.5),
    optimizer='SGD',
    optimizer_params={'learning_rate': learning_rate},
    num_epoch=n_epoch,
    batch_end_callback=None,
    epoch_end_callback=None
)

for k in module.get_params():
    print(k)

