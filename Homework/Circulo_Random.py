# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:01:10 2018

@author: Cortana
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
import numpy as np

# radius of the circle
circle_r = 3
circle_r2 = 30
# center of the circle (x, y)
circle_x = 0
circle_y = 0

alpha = np.zeros([100,1],dtype=np.float64)
r = np.zeros([100,1],dtype=np.float64)
x1 = np.zeros([100,1],dtype=np.float64)
y1 = np.zeros([100,1],dtype=np.float64)
alphax = np.zeros([100,1],dtype=np.float64)
rr = np.zeros([100,1],dtype=np.float64)
xx = np.zeros([100,1],dtype=np.float64)
yy = np.zeros([100,1],dtype=np.float64)
cont=0
for i in range(100):
    
    # random angle
    alpha[i] = 2 * math.pi * random.random()
    # random radius
    r[i] = circle_r * random.random()
    # calculating coordinates
    x1[i] = r[i] * math.cos(alpha[i]) + circle_x
    y1[i] = r[i] * math.sin(alpha[i]) + circle_y
    
    #cálculo del círculo exterior
    alphax[i] = 2 * math.pi * random.random()
    # random radius
    rr[i] = circle_r2 * random.random()
    # calculating coordinates
    xx[i] = rr[i] * math.cos(alphax[i]) 
    yy[i] = rr[i] * math.sin(alphax[i]) 
    
    if (xx[i]<1):
        xx[i] = xx[i] - 3
    else:
        xx[i] = xx[i] + 3  
        
    if (yy[i]<1):
        yy[i] = yy[i] - 3
    else:
        yy[i] = yy[i] + 3   
    cont=cont+1
    
            

plt.plot(x1,y1, 'ro', xx, yy, 'bo')
plt.ylabel('some numbers')
plt.show()
#print("Random point", (x, y)) x_train=np.zeros([21,1],dtype=np.float64)
yp1=np.ones([100,1],dtype=np.float64)
yp2=np.zeros([100,1],dtype=np.float64)

#Aquí empieza el entrenamiento de la red
xaux1 = np.column_stack((x1,y1))
xaux2 = np.column_stack((xx,yy))
x_train = np.concatenate((xaux1, xaux2), axis=0)
y_train = np.concatenate((yp1, yp2), axis=0)

n_samples=cont*2 

x_train=x_train.tolist()
y_train=y_train.tolist()
#Model parameters
Wco=tf.Variable(tf.random_uniform((2,20),-1,1)) #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bco=tf.Variable(tf.random_uniform((20,),-1,1))

Wcs=tf.Variable(tf.random_uniform((20,1),-1,1)) #Genero valores aleatorios para los pesos, con 2 neuronas de entrada, 10 nodos en la capa oculta y números entre -1 y 1
bcs=tf.Variable(tf.random_uniform((1,),-1,1))

#M=odel input and output
x = tf.placeholder(tf.float32,(n_samples,2), name='X') #Tamaño de los tensores de entrada
y = tf.placeholder(tf.float32,(n_samples,1), name='Y')

OutputCo = tf.tanh(tf.matmul(x,Wco)+bco)
Output = tf.sigmoid(tf.matmul(OutputCo,Wcs)+bcs)

#Loss
loss = tf.reduce_sum(tf.square(Output - y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

tf.summary.scalar('loss',loss)##
merged = tf.summary.merge_all() ##

#Training loop
init = tf.global_variables_initializer()
sess = tf.Session()

train_writer = tf.summary.FileWriter( "./Num_Circles/1" , sess.graph)

sess.run(init)
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})
    loss_tb, summary_tb = sess.run([loss,merged],{x: x_train, y: y_train})
    train_writer.add_summary(summary_tb, i)

train_writer.close() ##
#Evaluate training accuracy
curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output = sess.run([Wco,bco,Wcs,bcs,loss,Output],{x: x_train, y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s Loss: %s Output: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs, curr_loss, curr_Output))
