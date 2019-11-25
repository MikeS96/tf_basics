import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import csv

#data=np.zeros([21,2],dtype=np.float64)
x_train=np.zeros([21,1],dtype=np.float64)
y_train=np.zeros([21,1],dtype=np.float64)
cont=0;

for i in range(0,21):
    x_train[cont] = (math.pi)*i*0.1
    y_train[cont] = math.sin(x_train[cont] )
    cont=cont+1
          
n_samples=cont 
print(n_samples)   

X = tf.placeholder(tf.float32,(n_samples,1), name='X')
Y = tf.placeholder(tf.float32,(n_samples,1), name='Y')

# Model parameters
Wco = tf.Variable(tf.random_uniform((1,10),-1,1))
bco = tf.Variable(tf.random_uniform((10,),-1,1))

Wcs = tf.Variable(tf.random_uniform((10,1),-1,1))
bcs = tf.Variable(tf.random_uniform((1,),-1,1))

OutputCo= tf.tanh(tf.matmul(X,Wco)+bco)
Output= (tf.matmul(OutputCo,Wcs)+bcs)


#loss = tf.reduce_sum(tf.square(Y-Output)) # sum of the squares
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = tf.train.MomentumOptimizer(0.01,0.9)

train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init) 

for i in range(1):
  sess.run(train, {X: x_train, Y: y_train})

curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss, curr_Output = sess.run([Wco, bco,Wcs, bcs, loss,Output], {X:  x_train, Y: y_train})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss))

#Xg, Yg = data.T[0], data.T[1]
Xg, Yg = x_train, y_train
plt.plot(Xg, Yg, 'bo', label='Datos Deseados')
plt.plot(Xg, curr_Output, 'r*', label='Salida Red')
plt.legend()
plt.show()


for i in range(10000):
  sess.run(train, {X: x_train, Y: y_train})

curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss, curr_Output = sess.run([Wco, bco,Wcs, bcs, loss,Output], {X:  x_train, Y: y_train})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss))

Xg, Yg = x_train, y_train
plt.plot(Xg, Yg, 'bo', label='Datos Deseados')
plt.plot(Xg, curr_Output, 'r*', label='Salida Red')
plt.legend()
plt.show()
