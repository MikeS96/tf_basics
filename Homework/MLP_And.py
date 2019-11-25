# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 02:12:28 2018

@author: Cortana
"""

import tensorflow as tf


#training Data
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[1],[1],[1]]

#Model parameters
Wco=tf.Variable(tf.random_uniform((2,10),-1,1),name="Wco") #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bco=tf.Variable(tf.random_uniform((10,),-1,1),name="bco")

Wcs=tf.Variable(tf.random_uniform((10,1),-1,1),name="Wcs") #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bcs=tf.Variable(tf.random_uniform((1,),-1,1),name="bcs")

#M=odel input and output
x = tf.placeholder(tf.float32,(4,2),name="x") #Tamaño de los tensores de entrada
y = tf.placeholder(tf.float32,(4,1),name="y")

OutputCo = tf.tanh(tf.matmul(x,Wco)+bco)
Output = tf.sigmoid(tf.matmul(OutputCo,Wcs)+bcs)

#Loss
loss = tf.reduce_sum(tf.square(Output - y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

tf.summary.scalar('loss',loss)##
merged = tf.summary.merge_all() ##

#Training loop
init = tf.global_variables_initializer()
sess = tf.Session()

train_writer = tf.summary.FileWriter("./MLP/1",sess.graph)

sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    loss_tb, summary_tb = sess.run([loss,merged],{x: x_train, y: y_train})
    train_writer.add_summary(summary_tb, i)

train_writer.close() ##
#Evaluate training accuracy
curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output = sess.run([Wco,bco,Wcs,bcs,loss,Output],{x: x_train, y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s Loss: %s Output: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output))