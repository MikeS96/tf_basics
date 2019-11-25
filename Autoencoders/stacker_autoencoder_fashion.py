# -*- coding: utf-8 -*-
"""
Created on Fri May  4 02:51:10 2018

@author: Cortana
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import itertools
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion/', one_hot=True)


#Primer autoencoder apilado

X1 = tf.placeholder(tf.float32,(None, None), name='X1')

Wco1 = tf.Variable(tf.random_uniform((784,300),-1,1))
bco1 = tf.Variable(tf.random_uniform((300,),-1,1))

Wcs1 = tf.Variable(tf.random_uniform((300,784),-1,1))
bcs1 = tf.Variable(tf.random_uniform((784,),-1,1))

OutputCo1= tf.nn.sigmoid(tf.matmul(X1,Wco1)+bco1)


OutputAE1 = tf.nn.sigmoid((tf.matmul(OutputCo1,Wcs1)+bcs1))

lossAE1 = tf.reduce_mean(tf.square(OutputAE1-X1))

optimizerAE1 = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

trainAE1 = optimizerAE1.minimize(lossAE1)

#Segundo autoencoder apilado

X2 = tf.placeholder(tf.float32,(None, None), name='X2')

Wco2 = tf.Variable(tf.random_uniform((300,100),-1,1))
bco2 = tf.Variable(tf.random_uniform((100,),-1,1))

Wcs2 = tf.Variable(tf.random_uniform((100,300),-1,1))
bcs2 = tf.Variable(tf.random_uniform((300,),-1,1))

OutputCo2= tf.nn.sigmoid(tf.matmul(X2,Wco2)+bco2)


OutputAE2 = tf.nn.sigmoid((tf.matmul(OutputCo2,Wcs2)+bcs2))

lossAE2 = tf.reduce_mean(tf.square(OutputAE2-X2))

optimizerAE2 = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

trainAE2 = optimizerAE2.minimize(lossAE2)

#Tercer autoencoder apilado

X3 = tf.placeholder(tf.float32,(None, None), name='X3')

Wco3 = tf.Variable(tf.random_uniform((100,50),-1,1))
bco3 = tf.Variable(tf.random_uniform((50,),-1,1))

Wcs3 = tf.Variable(tf.random_uniform((50,100),-1,1))
bcs3 = tf.Variable(tf.random_uniform((100,),-1,1))

OutputCo3= tf.nn.sigmoid(tf.matmul(X3,Wco3)+bco3)
print(type(OutputCo3))

OutputAE3 = tf.nn.sigmoid((tf.matmul(OutputCo3,Wcs3)+bcs3))

lossAE3 = tf.reduce_mean(tf.square(OutputAE3-X3))

optimizerAE3 = tf.train.AdagradOptimizer(0.1,0.001)

trainAE3 = optimizerAE3.minimize(lossAE3)

#Capa clasificadora   

X4 = tf.placeholder(tf.float32,(None, None), name='X4')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

#Wco4 = tf.Variable(tf.random_uniform((100,50),-1,1))
#bco4 = tf.Variable(tf.random_uniform((50,),-1,1))

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


for i in range(20000):
    batch_xs, batch_ys = data.train.next_batch(50)
    sess.run(trainAE1, feed_dict={X1: batch_xs})

for i in range(20000):
    batch_xs, batch_ys = data.train.next_batch(50)
    np_OutputCo1, curr_lossAE1  = sess.run([OutputCo1,lossAE1], feed_dict={X1: batch_xs})
    sess.run(trainAE2, feed_dict={X2: np_OutputCo1})
    
for i in range(20000):
    batch_xs, batch_ys = data.train.next_batch(50)
    np_OutputCo1, curr_lossAE1  = sess.run([OutputCo1,lossAE1], feed_dict={X1: batch_xs})
    np_OutputCo2, curr_lossAE2  = sess.run([OutputCo2,lossAE2], feed_dict={X2: np_OutputCo1})
    sess.run(trainAE3, feed_dict={X3: np_OutputCo2})

for i in range(20000):
    batch_xs, batch_ys = data.train.next_batch(50)
    np_OutputCo1, curr_lossAE1  = sess.run([OutputCo1,lossAE1], feed_dict={X1: batch_xs})
    np_OutputCo2, curr_lossAE2  = sess.run([OutputCo2,lossAE2], feed_dict={X2: np_OutputCo1})
    np_OutputCo3, curr_lossAE3  = sess.run([OutputCo3,lossAE3], feed_dict={X3: np_OutputCo2})
    sess.run(trainCS, feed_dict={X4: np_OutputCo3, Y: batch_ys})
    
Out_CM=np.zeros([10000,1],dtype=np.float64)  
y_test_CM=np.zeros([10000,1],dtype=np.float64)
Out_test_CM=np.zeros([10000,1],dtype=np.float64)

curr_lossAE1 = sess.run(lossAE1, feed_dict={X1: data.test.images})
np_OutputCo1 = sess.run(OutputCo1, feed_dict={X1: data.test.images})
np_OutputCo2 = sess.run(OutputCo2, feed_dict={X2: np_OutputCo1})
np_OutputCo3 = sess.run(OutputCo3, feed_dict={X3: np_OutputCo2})
curr_loss, curr_Output = sess.run([lossCS,OutputCS], {X4 : np_OutputCo3, Y :data.test.labels})

print("Curr loss: ", curr_loss)

#correct_prediction = tf.equal(tf.argmax(curr_Output, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(sess.run(accuracy, feed_dict={X: mnist.test.images,
#                                      Y: mnist.test.labels}))
    
for ii in range(0,10000):
    Out_test_CM[ii] = np.argmax(curr_Output[ii,:])
    y_test_CM[ii] = np.argmax(data.test.labels[ii,:])
    
  
class_names=['T-Shirt', 'Trouser', 'Pullover','Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag','Ankle boot']    

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


correct_prediction = tf.equal(tf.argmax(OutputCS, 1),tf.argmax(data.test.labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


np_OutputCo1 = sess.run(OutputCo1, feed_dict={X1: data.test.images})
np_OutputCo2 = sess.run(OutputCo2, feed_dict={X2: np_OutputCo1})
np_OutputCo3 = sess.run(OutputCo3, feed_dict={X3: np_OutputCo2})

print("Accuracy: ",sess.run(accuracy, feed_dict={X4: np_OutputCo3,Y: data.test.labels}))
