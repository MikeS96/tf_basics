
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#X = tf.placeholder(tf.float32, [None, 784])
#Y = tf.placeholder(tf.float32, [None, 10])

X = tf.placeholder(tf.float32,(None, None), name='X')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

x_image = tf.reshape(X, [-1, 28, 28, 1])

W_conv1 =tf.Variable(tf.truncated_normal([5, 5, 1, 32],stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
out_conv1=tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(out_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv2 =tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
out_conv2=tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(out_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_fc1 =tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024],stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 =tf.Variable(tf.truncated_normal([1024, 10],stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss= tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
optimizer  = tf.train.AdamOptimizer(1e-4)

train = optimizer.minimize(loss)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(" Matrix de confusion Normalizada ")
    else:
        print('Matrix de confusion No Normalizada')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Salida Deseada')
    plt.xlabel('Salida Estimada')


with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

#graph_location = tempfile.mkdtemp()
#print('Saving graph to: %s' % graph_location)
#train_writer = tf.summary.FileWriter(graph_location)
#train_writer.add_graph(tf.get_default_graph())

#with tf.Session() as sess:
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(500):
    batch = mnist.train.next_batch(50)
    if i % 50 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            X: batch[0], Y: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})

#print('test accuracy %g' % accuracy.eval(feed_dict={
#        X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))


    # Test trained model
Out_CM=np.zeros([10000,1],dtype=np.float64)  
y_test_CM=np.zeros([10000,1],dtype=np.float64)
Out_test_CM=np.zeros([10000,1],dtype=np.float64)
curr_loss,curr_Output= sess.run([loss,y_conv], feed_dict={X:mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
for ii in range(0,10000):
    Out_test_CM[ii] = np.argmax(curr_Output[ii,:])
    y_test_CM[ii] = np.argmax(mnist.test.labels[ii,:])
    
  
class_names=['0', '1', '2','3', '4', '5','6', '7', '8','9']    


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_CM, Out_test_CM)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matrix de confusion No Normalizada')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Matrix de confusion Normalizada')

plt.show()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ",sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels, keep_prob: 1.0}))
