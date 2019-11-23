# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:47:18 2018

@author: Anastasia
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

# Ruta de los archivos con los datos de entrenamiento y validación
DATA_FILE_DATA = 'data/Motor_Data.csv'
DATA_FILE_TRAINING = 'data/Motor_Training_Data.csv'
DATA_FILE_TEST = 'data/Motor_Vali_Data.csv' 
DATA_FILE_PESOS = 'data/MLP_Pesos.csv'

#Definición de los arreglos que almacenarán los datos de la red para obtener el mínimo y máximo
data_raw=np.zeros([167,5],dtype=np.float64)
x_train_raw=np.zeros([167,4],dtype=np.float64)
y_train_raw=np.zeros([167,1],dtype=np.float64)

#Lectura de los datos para obtener el mínimo y máximo para la normalización
cont=0;
with open(DATA_FILE_DATA) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data_raw[cont,:]=(np.asarray(row))  
        x_train_raw[cont,0] = data_raw[cont,0]
        x_train_raw[cont,1] = data_raw[cont,1]        
        x_train_raw[cont,2] = data_raw[cont,2]
        x_train_raw[cont,3] = data_raw[cont,3]
        y_train_raw[cont,0] = data_raw[cont,4]
                      
        cont=cont+1

#Normalización de los datos de entrada

x1_maximo=max(x_train_raw[:,0])
x1_minimo=min(x_train_raw[:,0])
x2_maximo=max(x_train_raw[:,1])
x2_minimo=min(x_train_raw[:,1])
x3_maximo=max(x_train_raw[:,2])
x3_minimo=min(x_train_raw[:,2])
x4_maximo=max(x_train_raw[:,3])
x4_minimo=min(x_train_raw[:,3])

#Normalización de los datos de salida

y_maximo=max(y_train_raw)
y_minimo=min(y_train_raw)
      
#Definición de los arreglos que almacenarán los datos de entrenamiento
data=np.zeros([117,5],dtype=np.float64)
x_train=np.zeros([117,4],dtype=np.float64)
y_train=np.zeros([117,1],dtype=np.float64)
miguelon=np.zeros([117,1],dtype=np.float64)

#Definición de arreflos que contendrán los datos de entrenamiento normalizados
x_trainN=np.zeros([117,4],dtype=np.float64)
y_trainN=np.zeros([117,1],dtype=np.float64)
#Lectura de los datos de entrenamiento desde el archivo
cont=0;
with open(DATA_FILE_TRAINING) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data[cont,:]=(np.asarray(row))  
        x_train[cont,0] = data[cont,0]
        x_train[cont,1] = data[cont,1]        
        x_train[cont,2] = data[cont,2]
        x_train[cont,3] = data[cont,3]
        y_train[cont,0] = data[cont,4]
        miguelon[cont,0] = cont
                      
        cont=cont+1

n_samples=cont 
print(n_samples)   

#Proceso de normalización de los datos de entrada
for i in range(0,117):
    x_trainN[i,0] = (2*((x_train[i,0]-x1_minimo)/(x1_maximo-x1_minimo)))+(-1)
    x_trainN[i,1] = (2*((x_train[i,1]-x2_minimo)/(x2_maximo-x2_minimo)))+(-1)
    x_trainN[i,2] = (2*((x_train[i,2]-x3_minimo)/(x3_maximo-x3_minimo)))+(-1)
    x_trainN[i,3] = (2*((x_train[i,3]-x4_minimo)/(x4_maximo-x4_minimo)))+(-1)
    y_trainN[i] = (2*((y_train[i]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    
x_train=x_trainN  
y_train=y_trainN    

# Definición de los place holders para los datos de entrenamiento y validación
# como se tiene una cantidad de datos diferentes para entrenar (120)  y para validar #(30)  se deja indefinido el tamaño 
X = tf.placeholder(tf.float32,(None, None), name='X')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

# Definición de las variables para los pesos de la RNA
Wco = tf.Variable(tf.random_uniform((4,10),-1,1), name='Wco')
bco = tf.Variable(tf.random_uniform((10,),-1,1), name='bco')

Wcs = tf.Variable(tf.random_uniform((10,1),-1,1), name='Wcs')
bcs = tf.Variable(tf.random_uniform((1,),-1,1), name='bcs')

#Calculo de la salida.
#Capa oculta tangente sigmoidal
#Capa de salida sigmoidal
OutputCo= tf.tanh(tf.matmul(X,Wco)+bco)
Output=(tf.matmul(OutputCo,Wcs)+bcs)

loss = tf.reduce_sum(tf.square(Y-Output)) # sum of the squares
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.01) #Este asi tambien puede funcionar, 25000 epocas
#optimizer = tf.train.MomentumOptimizer(0.01,0.9) #Con 25000 epocas da re melo, estos parámetros y 10 nodos
#optimizer = tf.train.MomentumOptimizer(0.01,0.9,use_nesterov=True)
#optimizer = tf.train.AdagradOptimizer(0.1)
optimizer = tf.train.RMSPropOptimizer(0.001,0.9,0.3,1e-10) #(0.001,0.9,0.3,1e-10) y con 0.01 da melo, 10000 epocas
#optimizer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

train = optimizer.minimize(loss)

#tf.summary.scalar('Loss', loss)  #Pendejadas de Tensorboard
#merged = tf.summary.merge_all()

# training loop
init = tf.global_variables_initializer()


sess = tf.Session()


#train_writer = tf.summary.FileWriter( "./Momentum_N/0.01_0.9_10_25000" , sess.graph) #Más pendejadas de Tensorboard

sess.run(init) 

for i in range(1):
  sess.run(train, {X: x_train, Y: y_train})
  

curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss, curr_Output = sess.run([Wco, bco,Wcs, bcs, loss,Output], {X:  x_train, Y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss))

Xg, Yg = x_train, y_train
plt.plot(miguelon, Yg, 'bo', label='Datos Deseados')
plt.plot(miguelon, curr_Output, 'r*', label='Salida Red')
plt.legend()
plt.show()


for i in range(50000):
  sess.run(train, {X: x_train, Y: y_train})
  #loss_tb, summary_tb = sess.run([loss,merged],{X: x_train, Y: y_train})
  #train_writer.add_summary(summary_tb, i)
    
    
#train_writer.close()

curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss, curr_Output = sess.run([Wco, bco,Wcs, bcs, loss,Output], {X:  x_train, Y: y_train})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss))

Xg, Yg = x_train, y_train
plt.plot(miguelon, Yg, 'bo', label='Datos Deseados')
plt.plot(miguelon, curr_Output, 'r*', label='Salida Red')
plt.legend()
plt.show()

#Desnormalización de los datos de salida
curr_OutputN=np.zeros([117,1],dtype=np.float64)

for i in range(0,117):
    curr_OutputN[i,0] = ((y_maximo-y_minimo)*((curr_Output[i,0]+1)/2))+y_minimo
        
#Plotteo de los datos desnormalizados
Xg, Yg = x_train, y_train
plt.plot(miguelon, curr_OutputN, 'g', label='Salida Red')
plt.legend("Salida Desnormalizada")
plt.show()

#Definición de los arreglos que almacenaráan los datos de validación
data2=np.zeros([50,5],dtype=np.float64)
x_test=np.zeros([50,4],dtype=np.float64)
y_test=np.zeros([50,1],dtype=np.float64)
josefina=np.zeros([50,1],dtype=np.float64)

#Definición de arreflos que contendrán los datos de validación normalizados
x_testN=np.zeros([50,4],dtype=np.float64)
y_testN=np.zeros([50,1],dtype=np.float64)
#Lectura de los datos de entrenamiento desde el archivo
cont=0;
with open(DATA_FILE_TEST) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data2[cont,:]=(np.asarray(row))  
        x_test[cont,0] = data2[cont,0]
        x_test[cont,1] = data2[cont,1]        
        x_test[cont,2] = data2[cont,2]
        x_test[cont,3] = data2[cont,3]
        y_test[cont,0] = data2[cont,4]
        josefina[cont,0] = cont
                      
        cont=cont+1

n_samples=cont 
print(n_samples) 

#Proceso de normalización de los datos de entrada
for i in range(0,50):
    x_testN[i,0] = (2*((x_test[i,0]-x1_minimo)/(x1_maximo-x2_minimo)))+(-1)
    x_testN[i,1] = (2*((x_test[i,1]-x2_minimo)/(x2_maximo-x2_minimo)))+(-1)
    x_testN[i,2] = (2*((x_test[i,2]-x3_minimo)/(x2_maximo-x2_minimo)))+(-1)
    x_testN[i,3] = (2*((x_test[i,3]-x4_minimo)/(x2_maximo-x2_minimo)))+(-1)
    y_testN[i] = (2*((y_test[i]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    
x_test=x_testN   
y_test=y_testN 

curr_loss2, curr_Output2 = sess.run([loss,Output], {X:  x_test, Y: y_test})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("loss2: %s"%(curr_loss2))

#Xg, Yg = x_train, y_train
#plt.plot(miguelon, Yg, 'bo', label='Datos Deseados de Entrenamiento')
#plt.plot(miguelon, curr_Output, 'r*', label='Salida Red con Datos de Entrenamiento')
plt.plot(josefina, y_test, 'yo', label='Datos Deseados de Validacion')
plt.plot(josefina, curr_Output2, 'g*', label='Salida Red con Datos de Validación')
plt.legend()
plt.show()

plt.plot(josefina, y_test, 'r', label='Datos Deseados de Validacion')
plt.plot(josefina, curr_Output2, 'b', label='Salida Red con Datos de Validación')
plt.legend()
plt.show()

#Desnormalización de los datos de salida
curr_Output2N=np.zeros([50,1],dtype=np.float64)

for i in range(0,50):
    curr_Output2N[i,0] =((y_maximo-y_minimo)*((curr_Output2[i,0]+1)/2))+y_minimo
        
#Plotteo de los datos desnormalizados
plt.plot(josefina, curr_Output2N, 'g', label='Salida Red')
plt.legend("Salida Desnormalizada Datos validación")
plt.show()

#Aquí se envian los datos hacia el doc en excel
DATA_FILE_PESOS_CO = 'data/MLP_Pesos_CO.csv'
DATA_FILE_PESOS_CS = 'data/MLP_Pesos_CS.csv'

data3=np.zeros([10,5],dtype=np.float64)
data4=np.zeros([1,11],dtype=np.float64)

curr_loss,curr_Output,curr_Wco,curr_bco,curr_Wcs,curr_bcs = sess.run([loss,Output,Wco,bco,Wcs,bcs], {X:  x_train, Y: y_train})

print("WCO: %s BCO: %s "%(curr_Wco,curr_bco))
with open(DATA_FILE_PESOS_CO, 'w', newline='') as csvfile:
    weightwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for ii in range(0,10):
        data3[ii,0]=curr_Wco[0,ii]
        data3[ii,1]=curr_Wco[1,ii]
        data3[ii,2]=curr_Wco[2,ii]
        data3[ii,3]=curr_Wco[3,ii]
        data3[ii,4]=curr_bco[ii]
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