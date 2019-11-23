# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:28:20 2017

@author: Jose
"""

import tensorflow as tf

#train data
x_train=[[0,0],[0,1],[1,0],[1,1]]
y_train=[[0],[0],[0],[1]]

#Creacion del grafo
# Model Parameters
w=tf.Variable(tf.random_uniform((2,1),-1,1),name="w")
b=tf.Variable(tf.random_uniform((1,),-1,1),name="b")

# Model input and output
x=tf.placeholder(tf.float32,(4,2),name="x") #entrada de la red
y=tf.placeholder(tf.float32,(4,1),name="y") #salida deseada de la red
Output=tf.matmul(x,w)+b

#loss
loss= tf.reduce_sum(tf.square(Output-y))#sum of the squares
#optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)#algoritmo de entrenamiento
train=optimizer.minimize(loss)




#train loop

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log_Names/1",sess.graph)


for i in range(500):
    sess.run(train, {x:x_train, y:y_train})
    
    

    

#evaluate training accuaricy
curr_W, curr_b, curr_loss,curr_Output=sess.run([w,b,loss,Output],{x:x_train, y:y_train})
print("W: %s b: %s loss: %s Output: %s"%(curr_W,curr_b,curr_loss,curr_Output))

