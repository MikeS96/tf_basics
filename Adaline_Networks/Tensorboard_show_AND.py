# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:08:32 2018

@author: Cortana
"""

import tensorflow as tf

#training Data
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[0],[0],[1]]

#Model parameters
W=tf.Variable(tf.random_uniform((2,1),-1,1),name="W") #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
b=tf.Variable(tf.random_uniform((1,),-1,1),name="b")


#M=odel input and output
x = tf.placeholder(tf.float32,(4,2),name="x") #Tamaño de los tensores de entrada
y = tf.placeholder(tf.float32,(4,1),name="y")
Output = tf.matmul(x,W)+b

#Loss
loss = tf.reduce_sum(tf.square(Output - y)) #sumatoria de los cuadrados de la salida de la red - el valor deseado e=output-y

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) #Learning rate de la red, utilizada en el gradiente descendiente
train = optimizer.minimize(loss) #Descendent gradient

tf.summary.scalar('Loss',loss)
merged = tf.summary.merge_all()

#Training loop
init = tf.global_variables_initializer()
sess = tf.Session()

train_writer = tf.summary.FileWriter("./log_Scalar/1",sess.graph)

sess.run(init)


for i in range(500):
    sess.run(train, {x: x_train, y: y_train})
    loss_tb, summary_tb = sess.run([loss,merged],{x: x_train, y: y_train})
    train_writer.add_summary(summary_tb, i)


train_writer.close()

#Evaluate training accuracy
curr_W, curr_b, curr_loss, curr_Output = sess.run([W,b,loss,Output],{x: x_train, y: y_train})
print("W: %s b: %s Loss: %s Output: %s "%(curr_W, curr_b,curr_loss, curr_Output))
