# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 22:20:47 2018

@author: Cortana
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import itertools
from sklearn.metrics import confusion_matrix

import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epocas=30000
#Primer autoencoder apilado

X1 = tf.placeholder(tf.float32,(None, None), name='X1')

Wco1 = tf.Variable(tf.random_uniform((784,100),-1,1))
bco1 = tf.Variable(tf.random_uniform((100,),-1,1))

Wcs1 = tf.Variable(tf.random_uniform((100,784),-1,1))
bcs1 = tf.Variable(tf.random_uniform((784,),-1,1))

OutputCo1= tf.nn.sigmoid(tf.matmul(X1,Wco1)+bco1)
print(type(OutputCo1))

OutputAE1 = tf.nn.sigmoid((tf.matmul(OutputCo1,Wcs1)+bcs1))

lossAE1 = tf.reduce_mean(tf.square(OutputAE1-X1))

optimizerAE1 = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

trainAE1 = optimizerAE1.minimize(lossAE1)

#Segundo autoencoder apilado

X2 = tf.placeholder(tf.float32,(None, None), name='X2')

Wco2 = tf.Variable(tf.random_uniform((100,50),-1,1))
bco2 = tf.Variable(tf.random_uniform((50,),-1,1))



Wcs2 = tf.Variable(tf.random_uniform((50,100),-1,1))
bcs2 = tf.Variable(tf.random_uniform((100,),-1,1))

OutputCo2= tf.nn.sigmoid(tf.matmul(X2,Wco2)+bco2)
print(type(OutputCo2))

OutputAE2 = tf.nn.sigmoid((tf.matmul(OutputCo2,Wcs2)+bcs2))

lossAE2 = tf.reduce_mean(tf.square(OutputAE2-X2))

optimizerAE2 = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

trainAE2 = optimizerAE2.minimize(lossAE2)

#Capa clasificadora

X4 = tf.placeholder(tf.float32,(None, None), name='X4')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

#Wco4 = tf.Variable(tf.random_uniform((100,25),-1,1))
#bco4 = tf.Variable(tf.random_uniform((25,),-1,1))

Wcs4 = tf.Variable(tf.random_uniform((50,10),-1,1))
bcs4 = tf.Variable(tf.random_uniform((10,),-1,1))

#OutputCo4= tf.nn.tanh(tf.matmul(X4,Wco4)+bco4)



OutputCS = ((tf.matmul(X4,Wcs4)+bcs4))
lossCS= tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=OutputCS))

#OutputCS = tf.nn.sigmoid((tf.matmul(X4,Wcs4)+bcs4))
#lossCS= tf.reduce_mean(tf.square(Y-OutputCS))

optimizerCS = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

trainCS = optimizerCS.minimize(lossCS)

#Train

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()

for i in range(epocas):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(trainAE1, feed_dict={X1: batch_xs})
    
for i in range(epocas):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    np_OutputCo1, curr_lossAE1  = sess.run([OutputCo1,lossAE1], feed_dict={X1: batch_xs})
    sess.run(trainAE2, feed_dict={X2: np_OutputCo1})
    
for i in range(epocas):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    np_OutputCo1, curr_lossAE1  = sess.run([OutputCo1,lossAE1], feed_dict={X1: batch_xs})
    np_OutputCo2, curr_lossAE2  = sess.run([OutputCo2,lossAE2], feed_dict={X2: np_OutputCo1})
    sess.run(trainCS, feed_dict={X4: np_OutputCo2, Y: batch_ys})
    
Out_CM=np.zeros([10000,1],dtype=np.float64)  
y_test_CM=np.zeros([10000,1],dtype=np.float64)
Out_test_CM=np.zeros([10000,1],dtype=np.float64)

curr_lossAE1 = sess.run(lossAE1, feed_dict={X1: mnist.test.images})
np_OutputCo1 = sess.run(OutputCo1, feed_dict={X1: mnist.test.images})
np_OutputCo2 = sess.run(OutputCo2, feed_dict={X2: np_OutputCo1})
curr_loss, curr_Output = sess.run([lossCS,OutputCS], {X4 : np_OutputCo2, Y :mnist.test.labels})

print("Curr loss: ", curr_loss)

#correct_prediction = tf.equal(tf.argmax(curr_Output, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(sess.run(accuracy, feed_dict={X: mnist.test.images,
#                                      Y: mnist.test.labels}))
    
for ii in range(0,10000):
    Out_test_CM[ii] = np.argmax(curr_Output[ii,:])
    y_test_CM[ii] = np.argmax(mnist.test.labels[ii,:])
    
  
class_names=['0', '1', '2','3', '4', '5','6', '7', '8','9']    

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

correct_prediction = tf.equal(tf.argmax(OutputCS, 1),tf.argmax(mnist.test.labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


np_OutputCo1 = sess.run(OutputCo1, feed_dict={X1: mnist.test.images})
np_OutputCo2 = sess.run(OutputCo2, feed_dict={X2: np_OutputCo1})


print("Accuracy: ",sess.run(accuracy, feed_dict={X4: np_OutputCo2,Y: mnist.test.labels}))

      