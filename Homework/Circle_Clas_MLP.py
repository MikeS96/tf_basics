# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:33:08 2018

@author: Cortana
"""
import tensorflow as tf
import matplotlib.pyplot as plt


DATA_TB = 'Train_AND' ##

#training Data
x_train = [[-2,0],[-2.5,0],[0,2],[0,-2.5],[2,0],[2.5,0],[0,-2],[0,-2.5],[-1.5,1.5],[-3.5,3],[1.5,1.5],[3.5,3],[1.5,-1.5],[3.5,-3],[-1.5,-1.5],[-3.5,-3],[-1,-1],[-4,-3.7],[-1,1],[4,3.7]]
y_train = [[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]]
plt.plot([-2,2.5,0,0,2,2.5,0,0,-1.5,-3.5,1.5,3.5,1.5,3.5,-1.5,-3.5,-1,-4,-1,4], [0,0,2,-2.5,0,0,-2,-2.5,1.5,3,1.5,3,-1.5,-3,-1.5,-3,-1,-3.7,1,3.7,], 'ro')
plt.axis([-6, 6, -6, 6])
plt.show()
        
#Model parameters
Wco=tf.Variable(tf.random_uniform((2,5),-1,1)) #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bco=tf.Variable(tf.random_uniform((5,),-1,1))

Wcs=tf.Variable(tf.random_uniform((5,1),-1,1)) #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bcs=tf.Variable(tf.random_uniform((1,),-1,1))

#M=odel input and output
x = tf.placeholder(tf.float32,(20,2)) #Tamaño de los tensores de entrada
y = tf.placeholder(tf.float32,(20,1))

OutputCo = tf.tanh(tf.matmul(x,Wco)+bco)
Output = tf.sigmoid(tf.matmul(OutputCo,Wcs)+bcs)

#Loss
loss = tf.reduce_sum(tf.square(Output - y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.8)
train = optimizer.minimize(loss)

tf.summary.scalar('loss',loss)##
merged = tf.summary.merge_all() ##

#Training loop
init = tf.global_variables_initializer()
sess = tf.Session()

train_writer = tf.summary.FileWriter(DATA_TB , sess.graph) ##

sess.run(init)
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})
    loss_tb, summary_tb = sess.run([loss,merged],{x: x_train, y: y_train})
    train_writer.add_summary(summary_tb, i)

train_writer.close() ##
#Evaluate training accuracy
curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output = sess.run([Wco,bco,Wcs,bcs,loss,Output],{x: x_train, y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s Loss: %s Output: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output))