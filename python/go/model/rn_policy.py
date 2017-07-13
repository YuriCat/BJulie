# -*- coding: utf-8 -*-
# rn_policy.py
# Katsuki Ohto
# https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html
# http://kivantium.hateblo.jp/entry/2015/11/18/233834

import numpy as np
import tensorflow as tf

import go.go as go

#N_FILTERS = 128
#N_FILTERS = 192
N_FILTERS = 256
#N_FC_UNITS = 1024

# convolutional neural network
def make(x_placeholder):
    # モデルを作成する関数
    
    # 初期化関数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01, dtype = tf.float32)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape = shape, dtype = tf.float32)
        return tf.Variable(initial)

    # 畳み込み層
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

    # 畳み込み層1
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, go.IMAGE_PLAINS, N_FILTERS])
        b_conv1 = bias_variable([N_FILTERS])
        h_conv1 = tf.nn.relu(conv2d(x_placeholder, W_conv1) + b_conv1)
        #h_conv1 = tf.nn.relu(tf.nn.conv2d(x_placeholder, W_conv1, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv1)

    # 畳み込み層2
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv2 = bias_variable([N_FILTERS])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) + h_conv1

    # 畳み込み層3
    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv3 = bias_variable([N_FILTERS])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3) + h_conv2

    # 畳み込み層4
    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv4 = bias_variable([N_FILTERS])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4) + h_conv3

    # 畳み込み層5
    with tf.name_scope('conv5') as scope:
        W_conv5 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv5 = bias_variable([N_FILTERS])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5) + h_conv4
    
    # 畳み込み層6
    with tf.name_scope('conv6') as scope:
        W_conv6 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv6 = bias_variable([N_FILTERS])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6) + h_conv5
    
    # 畳み込み層7
    with tf.name_scope('conv7') as scope:
        W_conv7 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv7 = bias_variable([N_FILTERS])
        h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7) + h_conv6

    # 畳み込み層8
    with tf.name_scope('conv8') as scope:
        W_conv8 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv8 = bias_variable([N_FILTERS])
        h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8) + h_conv7

    # 畳み込み層9
    with tf.name_scope('conv9') as scope:
        W_conv9 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv9 = bias_variable([N_FILTERS])
        h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9) + h_conv8
    
    # 畳み込み層10
    with tf.name_scope('conv10') as scope:
        W_conv10 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv10 = bias_variable([N_FILTERS])
        h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10) + h_conv9

    # 畳み込み層11
    with tf.name_scope('conv11') as scope:
        W_conv11 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv11 = bias_variable([N_FILTERS])
        h_conv11 = tf.nn.relu(conv2d(h_conv10, W_conv11) + b_conv11) + h_conv10
    
    # 畳み込み層12
    with tf.name_scope('conv12') as scope:
        W_conv12 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv12 = bias_variable([N_FILTERS])
        h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12) + h_conv11

    # 畳み込み層13
    with tf.name_scope('conv12') as scope:
        W_conv13 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv13 = bias_variable([N_FILTERS])
        h_conv13 = tf.nn.relu(conv2d(h_conv12, W_conv13) + b_conv13) + h_conv12

    # 畳み込み層14
    with tf.name_scope('conv12') as scope:
        W_conv14 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv14 = bias_variable([N_FILTERS])
        h_conv14 = tf.nn.relu(conv2d(h_conv13, W_conv14) + b_conv14) + h_conv13

    # 畳み込み層15
    with tf.name_scope('conv12') as scope:
        W_conv15 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv15 = bias_variable([N_FILTERS])
        h_conv15 = tf.nn.relu(conv2d(h_conv14, W_conv15) + b_conv15) + h_conv14

    # 畳み込み層16
    with tf.name_scope('conv12') as scope:
        W_conv16 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv16 = bias_variable([N_FILTERS])
        h_conv16 = tf.nn.relu(conv2d(h_conv15, W_conv16) + b_conv16) + h_conv15

    # 畳み込み層17
    with tf.name_scope('conv12') as scope:
        W_conv17 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv17 = bias_variable([N_FILTERS])
        h_conv17 = tf.nn.relu(conv2d(h_conv16, W_conv17) + b_conv17) + h_conv16

    # 畳み込み層18
    with tf.name_scope('conv12') as scope:
        W_conv18 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv18 = bias_variable([N_FILTERS])
        h_conv18 = tf.nn.relu(conv2d(h_conv17, W_conv18) + b_conv18) + h_conv17

    # 畳み込み層19
    with tf.name_scope('conv12') as scope:
        W_conv19 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
        b_conv19 = bias_variable([N_FILTERS])
        h_conv19 = tf.nn.relu(conv2d(h_conv18, W_conv19) + b_conv19) + h_conv18

    # 最終層
    with tf.name_scope('last') as scope:
        W_convl = weight_variable([1, 1, N_FILTERS, 1])
        h_convl = conv2d(h_conv19, W_convl)
        h_flat = tf.reshape(h_convl, [-1, go.IMAGE_SIZE])
        b_last = bias_variable([go.IMAGE_SIZE])
        h_last = h_flat + b_last

    return h_last

def normalize(y):
    return tf.nn.softmax(y) # 確率配列を返す

def calc_loss(prob, labels):
    cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)))
    # TensorBoardで表示するよう指定
    #tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

def train(loss, learning_rate):
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_step

def count_correct_num(y, answer):
    correct_prediction = tf.equal(tf.argmax(y, 1), answer)
    return tf.reduce_sum(tf.to_int32(correct_prediction))
