# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:01:00 2018

@author: Cortana
"""

import tensorflow as tf

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

#Training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(500):
    sess.run(train, {x: x_train, y: y_train})
    
#Evaluate training accuracy
curr_W, curr_b, curr_loss, curr_Output = sess.run([W,b,loss,Output],{x: x_train, y: y_train})
print("W: %s b: %s Loss: %s Output: %s "%(curr_W, curr_b,curr_loss, curr_Output))