# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import load_data

Imgs, labels = load_data.getData()
trainX = load_data.preprocess(Imgs)

learning_rate = 0.1
training_epochs = 100
batch_size = 100
display_step = 1
examples_to_show = 10

X = tf.placeholder("float", [None, 64, 64, 1])


def encoder(img):
    input_layer = tf.reshape(img, [-1, 64, 64, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool3= tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)  

    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])
    dense1 = tf.layers.dense(inputs=pool3, units=1024, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4)

    # Logits Layer
    encoded = tf.layers.dense(inputs=encoder, units=30)
    return encoded

def decoder(x):
    # Convolutional Layer #4
    h1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=8*8*64, activation=tf.nn.relu)
    h3 = tf.reshape(h2, [-1, 8, 8, 64])
    conv4 = tf.layers.conv2d(
        inputs=h3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # upsampling Layer #4
    upsampling1= tf.image.resize_images(conv4, [16, 16])

    # Convolutional Layer #2 and Pooling Layer #2
    conv5 = tf.layers.conv2d(
        inputs=upsampling1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    upsampling2 = tf.image.resize_images(conv5, [32, 32])
    
    # Convolutional Layer #3 and Pooling Layer #3
    conv6 = tf.layers.conv2d(
        inputs=upsampling2,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    upsampling3 = tf.image.resize_images(conv6, [64, 64])

    decoded = tf.layers.conv2d(
        inputs=upsampling3,
        filters=1,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.sigmoid)
    
    return decoded

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
y_true = X

cost = tf.reduce_sum(tf.square(y_pred - y_true))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = 100
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = trainX[batch_size*i:batch_size*(i+1), :, :, :]
            #batch_ys = trainX[batch_size*epoch+i, :, :, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X:trainX[1100:2600]})
    
np.save("results.npy", encode_decode)
