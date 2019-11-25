#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 07:44:59 2018

@author: jose
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import itertools
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#build Autoencoder 1*************************************************
X1 = tf.placeholder(tf.float32,(None, None), name='X')
#Y1 = tf.placeholder(tf.float32,(None, None), name='Y')

Wco1 = tf.Variable(tf.random_uniform((784,500),-1,1))
bco1 = tf.Variable(tf.random_uniform((500,),-1,1))

Wcs1 = tf.Variable(tf.random_uniform((500,784),-1,1))
bcs1 = tf.Variable(tf.random_uniform((784,),-1,1))

OutputCo1= tf.nn.tanh(tf.matmul(X1,Wco1)+bco1)
print(type(OutputCo1))



OutputAE1=((tf.matmul(OutputCo1,Wcs1)+bcs1))


lossAE1= tf.reduce_mean(tf.square(OutputAE1-X1))

optimizerAE1  = tf.train.AdagradOptimizer(0.1,0.0001)

trainAE1 = optimizerAE1.minimize(lossAE1)



##build Autoencoder 2*************************************************
X2 = tf.placeholder(tf.float32,(None, None), name='X2')


Wco2 = tf.Variable(tf.random_uniform((500,300),-1,1))
bco2 = tf.Variable(tf.random_uniform((300,),-1,1))

Wcs2 = tf.Variable(tf.random_uniform((300,500),-1,1))
bcs2 = tf.Variable(tf.random_uniform((500,),-1,1))


OutputCo2= tf.nn.tanh(tf.matmul(X2,Wco2)+bco2)

OutputAE2=((tf.matmul(OutputCo2,Wcs2)+bcs2))


lossAE2= tf.reduce_mean(tf.square(OutputAE2-X2))

optimizerAE2  = tf.train.AdagradOptimizer(0.1,0.0001)

trainAE2 = optimizerAE2.minimize(lossAE2)


##build classifier layer*************************************************
X3 = tf.placeholder(tf.float32,(None, None), name='X3')
Y = tf.placeholder(tf.float32,(None,None), name='Y')

Wco3 = tf.Variable(tf.random_uniform((300,100),-1,1))
bco3 = tf.Variable(tf.random_uniform((100,),-1,1))

Wco4 = tf.Variable(tf.random_uniform((100,50),-1,1))
bco4 = tf.Variable(tf.random_uniform((50,),-1,1))

Wco5 = tf.Variable(tf.random_uniform((50,25),-1,1))
bco5 = tf.Variable(tf.random_uniform((25,),-1,1))


Wcs3 = tf.Variable(tf.random_uniform((25,10),-1,1))
bcs3 = tf.Variable(tf.random_uniform((10,),-1,1))


OutputCo3= tf.nn.relu(tf.matmul(X3,Wco3)+bco3)

OutputCo4= tf.nn.relu(tf.matmul(OutputCo3,Wco4)+bco4)

OutputCo5= tf.nn.relu(tf.matmul(OutputCo4,Wco5)+bco5)

OutputCls=((tf.matmul(OutputCo5,Wcs3)+bcs3))


lossCls= tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=OutputCls))

optimizerCls  = tf.train.AdagradOptimizer(0.1,0.0001)

trainCls = optimizerCls.minimize(lossCls)


#train*************************************************************************
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

  # Train
for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(trainAE1, feed_dict={X1: batch_xs})

np_OutputCo1=sess.run(OutputCo1, feed_dict={X1: batch_xs})

for _ in range(10000):
    sess.run(trainAE2, feed_dict={X2: np_OutputCo1})

np_OutputCo2=sess.run(OutputCo2, feed_dict={X2: np_OutputCo1})

for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(trainCls, feed_dict={X3: np_OutputCo2, Y: batch_ys})


Out_CM=np.zeros([10000,1],dtype=np.float64)  
y_test_CM=np.zeros([10000,1],dtype=np.float64)
Out_test_CM=np.zeros([10000,1],dtype=np.float64)

curr_lossAE1=sess.run(lossAE1, feed_dict={X1: mnist.test.images})
np_OutputCo1=sess.run(OutputCo1, feed_dict={X1: mnist.test.images})
np_OutputCo2=sess.run(OutputCo2, feed_dict={X2: np_OutputCo1})
curr_loss, curr_Output = sess.run([lossCls,lossCls], {X3 : np_OutputCo2, Y :mnist.test.labels})
#curr_loss,curr_Output= sess.run([lossCls,OutputCls], {X3:X_test, Y: y_test})

print("Curr loss: ", curr_loss)
  