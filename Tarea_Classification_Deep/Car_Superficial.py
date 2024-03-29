# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:49:15 2018

@author: Cortana
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv 
import itertools
from sklearn.metrics import confusion_matrix

# Ruta de los archivos con los datos de entrenamiento y validación
DATA_FILE_DATA = 'data/Cat_Data.csv'
DATA_FILE_TRAINING = 'data/Car_Training.csv'
DATA_FILE_TEST = 'data/Car_Test.csv' 

      
#Definición de los arreglos que almacenarán los datos de entrenamiento
data=np.zeros([1210,7],dtype=np.float64)
x_train=np.zeros([1210,6],dtype=np.float64)
y_train=np.zeros([1210,4],dtype=np.float64)
y_train_CM=np.zeros([1210,1],dtype=np.float64)
Out_CM=np.zeros([1210,1],dtype=np.float64)

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
        x_train[cont,4] = data[cont,4]
        x_train[cont,5] = data[cont,5]        
        y_train_CM[cont,0] = data[cont,6]
        #En archivo la clase viene como 0, 1 y 2. En este caso lo habitual es que quede una 
#neurona activa por clase. Es decir dependiendo de la clase solo se activará una #neurona
        if data[cont,6]==0:
               y_train[cont,0] =1
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =0
        if data[cont,6]==1:
               y_train[cont,0] =0
               y_train[cont,1] =1
               y_train[cont,2] =0
               y_train[cont,3] =0
        if data[cont,6]==2:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =1
               y_train[cont,3] =0 
        if data[cont,6]==3:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =1                      
        cont=cont+1
        
n_samples=cont 
print(n_samples)   

# Definición de los place holders para los datos de entrenamiento y validación
# como se tiene una cantidad de datos diferentes para entrenar (120)  y para validar #(30)  se deja indefinido el tamaño 
X = tf.placeholder(tf.float32,(None, None), name='X')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

# Definición de las variables para los pesos de la RNA
Wco = tf.Variable(tf.random_uniform((6,15),-1,1), name='Wco')
bco = tf.Variable(tf.random_uniform((15,),-1,1), name='bco')

Wcs = tf.Variable(tf.random_uniform((15,4),-1,1), name='Wcs')
bcs = tf.Variable(tf.random_uniform((4,),-1,1), name='bcs')

#Calculo de la salida.
#Capa oculta tangente sigmoidal
#Capa de salida sigmoidal
OutputCo= tf.nn.sigmoid(tf.matmul(X,Wco)+bco)
Output=tf.nn.sigmoid(tf.matmul(OutputCo,Wcs)+bcs)

#loss = tf.reduce_sum(tf.square(Y-Output)) # sum of the squares
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.01) #Este asi tambien puede funcionar, 25000 epocas
#optimizer = tf.train.MomentumOptimizer(0.01,0.9) #Con 25000 epocas da re melo, estos parámetros y 10 nodos
#optimizer = tf.train.MomentumOptimizer(0.01,0.9,use_nesterov=True)
#optimizer  = tf.train.AdagradOptimizer(0.1,0.0001)
optimizer = tf.train.RMSPropOptimizer(0.001,0.9,0.3,1e-10) #(0.001,0.9,0.3,1e-10) y con 0.01 da melo, 10000 epocas
#optimizer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

train = optimizer.minimize(loss)

tf.summary.scalar('Loss', loss)  #Pendejadas de Tensorboard
merged = tf.summary.merge_all()

# training loop
init = tf.global_variables_initializer()

sess = tf.Session()
train_writer = tf.summary.FileWriter( "./Momentum_N/0.01_0.9_10_25000" , sess.graph) #Más pendejadas de Tensorboard
sess.run(init) 

for i in range(10000):
  sess.run(train, {X: x_train, Y: y_train})
  loss_tb, summary_tb = sess.run([loss,merged],{X: x_train, Y: y_train})
  train_writer.add_summary(summary_tb, i)
 
    
train_writer.close()
curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Loss: %s Output: %s "%(curr_loss,curr_Output))


#Para la matriz de confusión se necesita la posición de la neurona que tuvo mayor #activación, esto determina la clase
for i in range(0,1210):
    Out_CM[i] = np.argmax(curr_Output[i,:]) #Neurona que obtuvo la salida ganadora

#Nombres de las clases para la matriz de confusión    
class_names=['Unacc', 'Acc', 'Good', 'V-Good']

#Función que permite graficar la matriz de confusión
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(" Matrix de confusion Normalizada ")
    else:
        print('Matrix de confusion No Normalizada')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Salida Deseada')
    plt.xlabel('Salida Estimada')

# Calculo de la matriz de confusión
cnf_matrix = confusion_matrix(y_train_CM, Out_CM)
np.set_printoptions(precision=2)

# Graficación de la matriz de confusión no normalizada 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matrix de confusion No Normalizada')

# Graficación de la matriz de confusión normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Matrix de confusion Normalizada')

plt.show()


#Definición de los arreglos que almacenaráan los datos de validación
data2=np.zeros([518,7],dtype=np.float64)
x_test=np.zeros([518,6],dtype=np.float64)
y_test=np.zeros([518,4],dtype=np.float64)
y_test_CM=np.zeros([518,1],dtype=np.float64)
Out_test_CM=np.zeros([518,1],dtype=np.float64)
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
        x_test[cont,4] = data2[cont,4]
        x_test[cont,5] = data2[cont,5]        
        y_test_CM[cont,0] = data2[cont,6]
        #En archivo la clase viene como 0, 1 y 2. En este caso lo habitual es que quede una 
#neurona activa por clase. Es decir dependiendo de la clase solo se activará una #neurona
        if data2[cont,6]==0:
               y_test[cont,0] =1
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =0
        if data2[cont,6]==1:
               y_test[cont,0] =0
               y_test[cont,1] =1
               y_test[cont,2] =0
               y_test[cont,3] =0
        if data2[cont,6]==2:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =1
               y_test[cont,3] =0 
        if data2[cont,6]==3:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =1                      
        cont=cont+1

n_samples=cont 
print(n_samples) 

curr_loss2, curr_Output2 = sess.run([loss,Output], {X:  x_test, Y: y_test})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("loss2: %s"%(curr_loss2))

for ii in range(0,518):
    Out_test_CM[ii] = np.argmax(curr_Output2[ii,:])
       
# Calculo de la matriz de confusión
cnf_matrix = confusion_matrix(y_test_CM, Out_test_CM)
np.set_printoptions(precision=2)

# Graficación de la matriz de confusión no normalizada 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matrix de confusion No Normalizada')

# Graficación de la matriz de confusión normalizada 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Matrix de confusion Normalizada')

plt.show()

#Impresion del Accuracy
current_prediction = tf.equal(tf.argmax(Output,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(current_prediction, tf.float32))
print("Test Accuracy= ",sess.run(accuracy, feed_dict={X:  x_test, Y: y_test}))
