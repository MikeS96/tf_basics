"""
Created on Sun Sep  3 18:23:01 2017
Aprendizaje de una superficie en TensorFlow con un MLP
@author: Jesús Lopez
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import tensorflow as tf

# Generación de los datos para la gráficación y para el entrenamiento
Xg = np.arange(-5, 5, 0.4)
Yg = np.arange(-5, 5, 0.4)
Xg, Yg = np.meshgrid(Xg, Yg)
Zg = (Xg**2 - Yg**2)

#Inicialización de Variables con los datos de entrenamiento
x_train=np.zeros([625,2],dtype=np.float64)
y_train=np.zeros([625,1],dtype=np.float64)

#Escritura de los datos de entrenamiento
x_train[:,0]=np.reshape(Xg,(625)) 
x_train[:,1]=np.reshape(Yg,(625)) 
y_train[:,0]=np.reshape(Zg,(625)) 

#Número de patrones de entrenamiento  
n_samples=625 

# Creacion de placeholders donde se pondrán los datos de entrenamiento
X = tf.placeholder(tf.float32,(n_samples,2), name='X')
Y = tf.placeholder(tf.float32,(n_samples,1), name='Y')

# Creación de la variables que tendrán los pesos y bias de la red
Wco = tf.Variable(tf.random_uniform((2,20),-1,1))
bco = tf.Variable(tf.random_uniform((20,),-1,1))

Wcs = tf.Variable(tf.random_uniform((20,1),-1,1))
bcs = tf.Variable(tf.random_uniform((1,),-1,1))

#Creación de los nodos que calcula de la salida de la red
OutputCo= tf.tanh(tf.matmul(X,Wco)+bco)
Output= (tf.matmul(OutputCo,Wcs)+bcs)

# Loss function MSE
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
#optimizer = tf.train.MomentumOptimizer(0.001,0.3)

train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init) 

#Se entrena la red por 10000 iteraciones
for i in range(1000):
  sess.run(train, {X: x_train, Y: y_train})

curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss, curr_Output = sess.run([Wco, bco,Wcs, bcs, loss,Output], {X:  x_train, Y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss))

#Figura donde se graficarán la superficie generada por la red
fig = plt.figure()
ax = fig.gca(projection='3d')
#Se convierte la salida de la red en matriz para la graficación de la superficie
Zg = np.reshape(curr_Output,(25,25))
surf = ax.plot_surface(Xg, Yg, Zg, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-30, 30)
plt.show()

#Figura donde se graficarán los puntos de entrenamiento
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Xgv=x_train[:,0]
Ygv=x_train[:,1]


ax.scatter(Xgv,Ygv,curr_Output,color='red',marker='*')
ax.scatter(Xgv,Ygv,y_train[:,0],color='blue',marker='o')

plt.show()

