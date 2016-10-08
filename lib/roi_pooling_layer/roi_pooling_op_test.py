#!/usr/bin/env python
# coding=utf-8
import numpy as np
import roi_pooling_op
import roi_pooling_op_grad
import tensorflow as tf


# import pdb


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


data = tf.convert_to_tensor(np.random.rand(32, 100, 100, 3), dtype=tf.float32)
rois = tf.convert_to_tensor(
    [[0, 10, 10, 20, 20], [10, 30, 30, 40, 40], [20, 30, 30, 40, 40], [30, 30, 30, 40, 40], [31, 30, 30, 40, 40]],
    dtype=tf.float32)

W = weight_variable([3, 3, 3, 1])
h = conv2d(data, W)
"""
五个参数分别代表
第一个参数是输入数据,一般是feature map
第二个参数分为五个子参数, 分别为  一个batch中的第几个数据,如0是第一个31是第32个等等  框的起始宽  框的起始高  框的结尾宽  框的结尾高
第三个参数为pooling后的高
第四个参数为pooling后的宽
第五个参数为缩放参数, 此参数意义为 上面输入的框是原图的比例 而现在输入的是feature map的数据 与原图相差几个卷积和pooling的差距 此参数是把框的大小调整为feature map的比例
    调整的大小为pooling层的数量 × pooling层的pooling尺寸（为什么不计算卷积层，因为卷积层一般为3*3 步长为1 这么计算的话 每个卷积层卷积后图像只损失2×2个像素，
    即使有10层卷积也就损失20*20个像素因此忽略不计  而pooling层的降维程度就很大了，如果pooling是2*2的，那么图像将直接缩小一半）
"""
[y, argmax] = roi_pooling_op.roi_pool(h, rois, 3, 3, 1.0 / 3)
# pdb.set_trace()
y_data = tf.convert_to_tensor(np.ones((5, 3, 3, 1)), dtype=tf.float32)
print y_data, y, argmax

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    # pdb.set_trace()
    for step in xrange(10):
        sess.run(train)
        print(step, sess.run(W))
        print(sess.run(y))
        # print(sess.run(argmax))

        # with tf.device('/gpu:0'):
        #  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
        #  print result.eval()
        # with tf.device('/cpu:0'):
        #  run(init)
