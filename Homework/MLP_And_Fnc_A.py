# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 02:12:28 2018

@author: Cortana
"""

import tensorflow as tf

DATA_TB = 'Train_AND' ##

#training Data
x_train = [[-1,-1],[-0.5,-0.5],[-1,0],[0,-1],[1,1],[0.5,0.5],[1,0],[0,1]]
y_train = [[0],[0],[0],[0],[1],[1],[1],[1]]

#Model parameters
Wco=tf.Variable(tf.random_uniform((2,8),-1,1)) #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bco=tf.Variable(tf.random_uniform((8,),-1,1))

Wcs=tf.Variable(tf.random_uniform((8,1),-1,1)) #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bcs=tf.Variable(tf.random_uniform((1,),-1,1))

#M=odel input and output
x = tf.placeholder(tf.float32,(8,2)) #Tamaño de los tensores de entrada
y = tf.placeholder(tf.float32,(8,1))

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

train_writer = tf.summary.FileWriter(DATA_TB , sess.graph) ##

sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    loss_tb, summary_tb = sess.run([loss,merged],{x: x_train, y: y_train})
    train_writer.add_summary(summary_tb, i)

train_writer.close() ##
#Evaluate training accuracy
curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output = sess.run([Wco,bco,Wcs,bcs,loss,Output],{x: x_train, y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s Loss: %s Output: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output))