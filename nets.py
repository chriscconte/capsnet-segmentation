from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import cv2


def squash(ss):
    ss_norm2 = F.sum(ss ** 2, axis=1, keepdims=True)
    """
    # ss_norm2 = F.broadcast_to(ss_norm2, ss.shape)
    # vs = ss_norm2 / (1. + ss_norm2) * ss / F.sqrt(ss_norm2): naive
    """
    norm_div_1pnorm2 = F.sqrt(ss_norm2) / (1. + ss_norm2)
    norm_div_1pnorm2 = F.broadcast_to(norm_div_1pnorm2, ss.shape)
    vs = norm_div_1pnorm2 * ss  # :efficient
    return vs


def get_norm(vs):
    return tf.reduce_sum(v_digit ** 2.0, axis=-1) ** 0.5  

init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)


class CapsNet(object):

    def __init__(self, use_reconstruction=False ):
        super(CapsNet, self).__init__()
        self.mmnist = mmnist
        self.n_iterations = 3  # dynamic routing
        self.n_grids = 10  # grid width of primary capsules layer
        self.n_raw_grids = self.n_grids
        self.use_reconstruction = use_reconstruction

        self.x = tf.placeholder(tf.float32, [None, 36, 36, 3])
        self.y = tf.placeholder(tf.float32, [None, 52, 3])

        x_composed, x_a, x_b = tf.split(self.x,num_or_size_splits=3,axis=3)
        y_composed, y_a, y_b = tf.split(self.y,num_or_size_splits=3,axis=2)

        valid_mask = (tf.reduce_sum(y_composed, axis=[1,2]) - 1.0) \
                      + * tf.ones_like(y_composed[:,0,0])

        vs, v = self.output(x_composed)

        x_rec_a = self.reconstruct(v * y_a)
        x_rec_b = self.reconstruct(v * y_b,reuse=True)
        loss_rec_a = tf.reduce_sum((x_rec_a - x_a) ** 2.0, axis=[1, 2, 3])
        loss_rec_b = tf.reduce_sum((x_rec_b - x_b) ** 2.0, axis=[1, 2, 3])

        self.loss_rec = (loss_rec_a + loss_rec_b) / 2.0
        self.x_recs = [x_rec_a,x_rec_b]
        self.x_sample = reconstruct(self.h_sample * self.y_sample[:, :, None], reuse=True)
        self.loss_cls = tf.reduce_sum(y_composed[:,:,0] * tf.maximum(0.0, 0.9 - length_v) ** 2.0
                                      + 0.5 * (1.0 - y_composed[:,:,0]) * tf.maximum(0.0, length_v - 0.1) ** 2.0,axis=-1)
        self.loss_cls = tf.reduce_sum(self.loss_cls*valid_mask)/tf.reduce_sum(valid_mask)
        self.loss_rec = tf.reduce_sum(self.loss_rec*valid_mask)/tf.reduce_sum(valid_mask)
        self.loss = self.loss_cls + 0.0005*self.loss_rec

        self.train = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=beta1).minimize(self.loss)

        if is_multi_mnist:
            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(length_v,tf.argmax(tf.squeeze(y_a), 1),k=2),tf.float32))+\
                            tf.reduce_mean(tf.cast(tf.nn.in_top_k(length_v,tf.argmax(tf.squeeze(y_b), 1),k=2),tf.float32))
            self.accuracy /= 2.0
            #this may be different from the paper
        else:
            correct_prediction = tf.equal(tf.argmax(y_composed[:,:,0], 1), tf.argmax(length_v, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def output(self, x):
        with tf.variable_scope('CapsNet',reuse=False):

            w_conv1 = tf.get_variable('w_conv1',[9,9,1,256],initializer=init)
            b_conv1 = tf.get_variable('w_conv1', [256], initializer=init)

            w_conv1 = tf.get_variable('w_conv2',[9,9,256,256],initializer=init)
            b_conv1 = tf.get_variable('b_conv2', [256], initializer=init)

            w_conv3 = tf.get_variable('wconv3',[9,9,256,8*32],initializer=init)
            b_conv3 = tf.get_variable('bconv3', [8*32], initializer=init)

            w_cap = tf.get_variable('wcap',[1,6,6,32,8,52,16],initializer=init)

            b = tf.get_variable('coupling_coefficient_logits',[1,6,6,32,1,10,1],initializer=tf.constant_initializer(0.0))
            self.conv1 = L.Convolution2D(1, 256, ksize=9, stride=1,
                                         initialW=init)
            self.conv2 = L.Convolution2D(256, 32 * 8, ksize=9, stride=2,
                                         initialW=init)
            self.Ws = chainer.ChainList(
                *[L.Convolution2D(8, 52 * 10, ksize=1, stride=1, initialW=init)
                  for i in range(32)])


        n_iters = self.n_iterations
        gg = self.n_grids * self.n_grids

        h1 = tf.nn.conv2d(x, w_conv1, [1, 1, 1, 1], padding='VALID') + b_conv1
        h1 = tf.maximum(h1, 0.05 * h1)
        
        h2 = tf.nn.conv2d(h1, w_conv2, [1, 1, 1, 1], padding='VALID') + b_conv2
        h2 = tf.maximum(h2, 0.05 * h2)

        h3 = tf.nn.conv2d(h2, w_conv3, [1, 2, 2, 1], padding='VALID') + b_conv3

        s_primary = tf.reshape(h3,[-1,6,6,32,8,1,1])
        v_primary = self.squash(s_primary,axis=4)

        u = v_primary
        u_ = tf.reduce_sum(u*wcap,axis=[4],keep_dims=True)
        s = tf.reduce_sum(u_*c,axis=[1,2,3],keep_dims=True)
        v = self.squash(s,axis=-1)

        # routing algo between PrimaryCaps and DigitCaps
        bs = self.xp.zeros((batchsize, 52, 32, gg), dtype='f')
        for i_iter in range(n_iters):
            b += tf.reduce_sum(u_*v,axis=-1,keep_dims=True)
            c =  tf.nn.softmax(b, dim=5)
            s = tf.reduce_sum(u_ * c, axis=[1, 2, 3], keep_dims=True)
            v = self.squash(s,axis=-1)

        # vs is the DigitCaps Layer
        vs_norm = get_norm(vs)

        return vs_norm, vs

    def reconstruct(self, vs, t):
        with tf.variable_scope('decoder',reuse=reuse):
            w_fc1 = tf.get_variable('wfc1',[16 * 52, 512],initializer=init)
            b_fc1 = tf.get_variable('bfc1',[512],initializer=init)

            w_fc2 = tf.get_variable('wfc2',[512, 1024],initializer=init)
            b_fc2 = tf.get_variable('bfc2',[1024],initializer=init)

            w_fc3 = tf.get_variable('wfc3',[1024, 1296],initializer=init)
            b_fc3 = tf.get_variable('bfc3',[1296],initializer=init)

        masked_vs = t * vs


        h1 = tf.nn.relu(tf.nn.add(tf.matmul(masked_vs, w_fc1) + b_fc1))
        h2 = tf.nn.relu(tf.nn.add(tf.matmul(h2, w_fc2) + b_fc2))
        x_recon = tf.nn.sigmoid(tf.nn.add(tf.matmul(h2, w_fc3) + b_fc3))

        x_recon  = tf.nn.reshape(x_recon, (-1, 1, 36, 36))

        return x_recon


