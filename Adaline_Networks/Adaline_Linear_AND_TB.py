# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 01:11:45 2018

@author: Cortana
"""

import tensorflow as tf

DATA_TB = 'Train_AND' ##
#training Data
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[0],[0],[1]]

#Model parameters
W=tf.Variable(tf.random_uniform((2,1),-1,1)) #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
b=tf.Variable(tf.random_uniform((1,),-1,1))


#M=odel input and output
x = tf.placeholder(tf.float32,(4,2)) #Tamaño de los tensores de entrada
y = tf.placeholder(tf.float32,(4,1))
Output = tf.matmul(x,W)+b

#Loss
loss = tf.reduce_sum(tf.square(Output - y)) #sumatoria de los cuadrados de la salida de la red - el valor deseado e=output-y

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) #Learning rate de la red, utilizada en el gradiente descendiente
train = optimizer.minimize(loss) #Descendent gradient

tf.summary.scalar('loss',loss)##
merged = tf.summary.merge_all() ##

#Training loop
init = tf.global_variables_initializer()
sess = tf.Session()

train_writer = tf.summary.FileWriter(DATA_TB , sess.graph) ##

sess.run(init)
for i in range(500):
    sess.run(train, {x: x_train, y: y_train})
    loss_tb, summary_tb = sess.run([loss,merged],{x: x_train, y: y_train})
    train_writer.add_summary(summary_tb, i)
    

train_writer.close() ##
#Evaluate training accuracy
curr_W, curr_b, curr_loss, curr_Output = sess.run([W,b,loss,Output],{x: x_train, y: y_train})
print("W: %s b: %s Loss: %s Output: %s "%(curr_W, curr_b,curr_loss, curr_Output))