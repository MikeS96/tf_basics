import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import csv

DATA_FILE_PESOS_CO = 'data/MLP_Pesos_CO.csv'
DATA_FILE_PESOS_CS = 'data/MLP_Pesos_CS.csv'

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


loss = tf.reduce_sum(tf.square(Y-Output)) # sum of the squares
#loss = tf.reduce_mean(tf.square(Y-Output))

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.001)
optimizer = tf.train.MomentumOptimizer(0.001,0.3)

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


for i in range(50000):
  sess.run(train, {X: x_train, Y: y_train})

curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss, curr_Output = sess.run([Wco, bco,Wcs, bcs, loss,Output], {X:  x_train, Y: y_train})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss))

Xg, Yg = x_train, y_train
plt.plot(Xg, Yg, 'bo', label='Datos Deseados')
plt.plot(Xg, curr_Output, 'r*', label='Salida Red')
plt.legend()
plt.show()

x_test=np.zeros([21,1],dtype=np.float64)
y_test=np.zeros([21,1],dtype=np.float64)
cont=0;

for i in range(0,21):
    x_test[cont] =(math.pi)*i*0.1+ ((math.pi)*0.05)
    y_test[cont] = math.sin(x_test[cont] )
    cont=cont+1
        
curr_loss2, curr_Output2 = sess.run([loss,Output], {X:  x_test, Y: y_test})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
#print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss))
 
Xg, Yg = x_train, y_train
plt.plot(Xg, Yg, 'bo', label='Datos Deseados de Entrenamiento')
plt.plot(Xg, curr_Output, 'r*', label='Salida Red con Datos de Entrenamiento')
plt.plot(x_test, y_test, 'yo', label='Datos Deseados de Validacion')
plt.plot(x_test, curr_Output2, 'g*', label='Salida Red con Datos de Validación')
plt.legend()
plt.show()

#DATA_FILE_PESOS_CO = 'data/MLP_Pesos_CO.csv'
#DATA_FILE_PESOS_CS = 'data/MLP_Pesos_CS.csv'

#Wco = tf.Variable(tf.random_uniform((1,10),-1,1))
#bco = tf.Variable(tf.random_uniform((10,),-1,1))
#
#Wcs = tf.Variable(tf.random_uniform((10,1),-1,1))
#bcs = tf.Variable(tf.random_uniform((1,),-1,1))

data3=np.zeros([10,2],dtype=np.float64)
data4=np.zeros([1,11],dtype=np.float64)

curr_loss,curr_Output,curr_Wco,curr_bco,curr_Wcs,curr_bcs = sess.run([loss,Output,Wco,bco,Wcs,bcs], {X:  x_train, Y: y_train})

print("WCO: %s BCO: %s "%(curr_Wco,curr_bco))
with open(DATA_FILE_PESOS_CO, 'w', newline='') as csvfile:
    weightwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for ii in range(0,10):
        data3[ii,0]=curr_Wco[0,ii]
        data3[ii,1]=curr_bco[ii]
        weightwriter.writerow(data3[ii,:]) 
        
print("WCS: %s BCS: %s "%(curr_Wcs,curr_bcs))        
with open(DATA_FILE_PESOS_CS, 'w', newline='') as csvfile:
    weightwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for ii in range(0,11):
        if ii<10:
            data4[0,ii]=curr_Wcs[ii,0]
        if ii==10:
            data4[0,ii]=curr_bcs[0]
    
    weightwriter.writerow(data4[0,:])         

