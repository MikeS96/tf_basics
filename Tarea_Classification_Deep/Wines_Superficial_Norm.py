# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:16:08 2018

@author: Cortana
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv 
import itertools
from sklearn.metrics import confusion_matrix

# Ruta de los archivos con los datos de entrenamiento y validación
DATA_FILE_DATA = 'data/wine_data.csv'
DATA_FILE_TRAINING = 'data/wine_training.csv'
DATA_FILE_TEST = 'data/wine_test.csv' 

#Definición de los arreglos que almacenarán los datos de la red para obtener el mínimo y máximo
data_raw=np.zeros([178,14],dtype=np.float64)
x_train_raw=np.zeros([178,13],dtype=np.float64)
y_train_raw=np.zeros([178,3],dtype=np.float64)

#Lectura de los datos para obtener el mínimo y máximo para la normalización
cont=0;
with open(DATA_FILE_DATA) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data_raw[cont,:]=(np.asarray(row))  
        x_train_raw[cont,0] = data_raw[cont,1]
        x_train_raw[cont,1] = data_raw[cont,2]        
        x_train_raw[cont,2] = data_raw[cont,3]
        x_train_raw[cont,3] = data_raw[cont,4]
        x_train_raw[cont,4] = data_raw[cont,5]
        x_train_raw[cont,5] = data_raw[cont,6]        
        x_train_raw[cont,6] = data_raw[cont,7]
        x_train_raw[cont,7] = data_raw[cont,8]
        x_train_raw[cont,8] = data_raw[cont,9]
        x_train_raw[cont,9] = data_raw[cont,10]        
        x_train_raw[cont,10] = data_raw[cont,11]
        x_train_raw[cont,11] = data_raw[cont,12]
        x_train_raw[cont,12] = data_raw[cont,13]
        #En archivo la clase viene como 0, 1 y 2. En este caso lo habitual es que quede una 
#neurona activa por clase. Es decir dependiendo de la clase solo se activará una #neurona
        if data_raw[cont,0]==1:
               y_train_raw[cont,0] =1
               y_train_raw[cont,1] =0
               y_train_raw[cont,2] =0                     
        if data_raw[cont,0]==2:
               y_train_raw[cont,0] =0
               y_train_raw[cont,1] =1
               y_train_raw[cont,2] =0 
        if data_raw[cont,0]==3:
               y_train_raw[cont,0] =0
               y_train_raw[cont,1] =0
               y_train_raw[cont,2] =1                       
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
x5_maximo=max(x_train_raw[:,4])
x5_minimo=min(x_train_raw[:,4])
x6_maximo=max(x_train_raw[:,5])
x6_minimo=min(x_train_raw[:,5])
x7_maximo=max(x_train_raw[:,6])
x7_minimo=min(x_train_raw[:,6])
x8_maximo=max(x_train_raw[:,7])
x8_minimo=min(x_train_raw[:,7])
x9_maximo=max(x_train_raw[:,8])
x9_minimo=min(x_train_raw[:,8])
x10_maximo=max(x_train_raw[:,9])
x10_minimo=min(x_train_raw[:,9])
x11_maximo=max(x_train_raw[:,10])
x11_minimo=min(x_train_raw[:,10])
x12_maximo=max(x_train_raw[:,11])
x12_minimo=min(x_train_raw[:,11])
x13_maximo=max(x_train_raw[:,12])
x13_minimo=min(x_train_raw[:,12])


#Normalización de los datos de salida

y_maximo=1
y_minimo=0
      
#Definición de los arreglos que almacenarán los datos de entrenamiento
data=np.zeros([126,14],dtype=np.float64)
x_train=np.zeros([126,13],dtype=np.float64)
y_train=np.zeros([126,3],dtype=np.float64)
y_train_CM=np.zeros([126,1],dtype=np.float64)
Out_CM=np.zeros([126,1],dtype=np.float64)

#Lectura de los datos de entrenamiento desde el archivo
cont=0;
with open(DATA_FILE_TRAINING) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data[cont,:]=(np.asarray(row))  
        x_train[cont,0] = data[cont,1]
        x_train[cont,1] = data[cont,2]        
        x_train[cont,2] = data[cont,3]
        x_train[cont,3] = data[cont,4]
        x_train[cont,4] = data[cont,5]
        x_train[cont,5] = data[cont,6]        
        x_train[cont,6] = data[cont,7]
        x_train[cont,7] = data[cont,8]
        x_train[cont,8] = data[cont,9]
        x_train[cont,9] = data[cont,10]        
        x_train[cont,10] = data[cont,11]
        x_train[cont,11] = data[cont,12]
        x_train[cont,12] = data[cont,13]
        y_train_CM[cont,0] = data[cont,0]
        #En archivo la clase viene como 0, 1 y 2. En este caso lo habitual es que quede una 
#neurona activa por clase. Es decir dependiendo de la clase solo se activará una #neurona
        if data[cont,0]==1:
               y_train[cont,0] =1
               y_train[cont,1] =0
               y_train[cont,2] =0                     
        if data[cont,0]==2:
               y_train[cont,0] =0
               y_train[cont,1] =1
               y_train[cont,2] =0 
        if data[cont,0]==3:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =1                       
        cont=cont+1

n_samples=cont 
print(n_samples)   
x_trainN=np.zeros([126,13],dtype=np.float64)
y_trainN=np.zeros([126,3],dtype=np.float64)
for i in range(0,126):
    x_trainN[i,0] = (2*((x_train[i,0]-x1_minimo)/(x1_maximo-x1_minimo)))+(-1)
    x_trainN[i,1] = (2*((x_train[i,1]-x2_minimo)/(x2_maximo-x2_minimo)))+(-1)
    x_trainN[i,2] = (2*((x_train[i,2]-x3_minimo)/(x3_maximo-x3_minimo)))+(-1)
    x_trainN[i,3] = (2*((x_train[i,3]-x4_minimo)/(x4_maximo-x4_minimo)))+(-1)
    x_trainN[i,4] = (2*((x_train[i,4]-x5_minimo)/(x5_maximo-x5_minimo)))+(-1)
    x_trainN[i,5] = (2*((x_train[i,5]-x6_minimo)/(x6_maximo-x6_minimo)))+(-1)
    x_trainN[i,6] = (2*((x_train[i,6]-x7_minimo)/(x7_maximo-x7_minimo)))+(-1)
    x_trainN[i,7] = (2*((x_train[i,7]-x8_minimo)/(x8_maximo-x8_minimo)))+(-1)
    x_trainN[i,8] = (2*((x_train[i,8]-x9_minimo)/(x9_maximo-x9_minimo)))+(-1)
    x_trainN[i,9] = (2*((x_train[i,9]-x10_minimo)/(x10_maximo-x10_minimo)))+(-1)
    x_trainN[i,10] = (2*((x_train[i,10]-x11_minimo)/(x11_maximo-x11_minimo)))+(-1)
    x_trainN[i,11] = (2*((x_train[i,11]-x12_minimo)/(x12_maximo-x12_minimo)))+(-1)
    x_trainN[i,12] = (2*((x_train[i,12]-x13_minimo)/(x13_maximo-x13_minimo)))+(-1)
    y_trainN[i,0] = (2*((y_train[i,0]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    y_trainN[i,1] = (2*((y_train[i,1]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    y_trainN[i,2] = (2*((y_train[i,2]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    
x_train=x_trainN  
y_train=y_trainN   
# Definición de los place holders para los datos de entrenamiento y validación
# como se tiene una cantidad de datos diferentes para entrenar (120)  y para validar #(30)  se deja indefinido el tamaño 
X = tf.placeholder(tf.float32,(None, None), name='X')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

# Definición de las variables para los pesos de la RNA
Wco = tf.Variable(tf.random_uniform((13,10),-1,1), name='Wco')
bco = tf.Variable(tf.random_uniform((10,),-1,1), name='bco')

Wcs = tf.Variable(tf.random_uniform((10,3),-1,1), name='Wcs')
bcs = tf.Variable(tf.random_uniform((3,),-1,1), name='bcs')

#Calculo de la salida.
#Capa oculta tangente sigmoidal
#Capa de salida sigmoidal
OutputCo= tf.tanh(tf.matmul(X,Wco)+bco)
Output=tf.sigmoid(tf.matmul(OutputCo,Wcs)+bcs)

#loss = tf.reduce_sum(tf.square(Y-Output)) # sum of the squares
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.01) #Este asi tambien puede funcionar, 25000 epocas
#optimizer = tf.train.MomentumOptimizer(0.01,0.9) #Con 25000 epocas da re melo, estos parámetros y 10 nodos
#optimizer = tf.train.MomentumOptimizer(0.01,0.9,use_nesterov=True)
optimizer  = tf.train.AdagradOptimizer(0.1,0.0001)
#optimizer = tf.train.RMSPropOptimizer(0.001,0.9,0.3,1e-10) #(0.001,0.9,0.3,1e-10) y con 0.01 da melo, 10000 epocas
#optimizer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-08)

train = optimizer.minimize(loss)

#tf.summary.scalar('Loss', loss)  #Pendejadas de Tensorboard
#merged = tf.summary.merge_all()

# training loop
init = tf.global_variables_initializer()


sess = tf.Session()

#train_writer = tf.summary.FileWriter( "./Momentum_N/0.01_0.9_10_25000" , sess.graph) #Más pendejadas de Tensorboard

sess.run(init) 

for i in range(20000):
  sess.run(train, {X: x_train, Y: y_train})
  #loss_tb, summary_tb = sess.run([loss,merged],{X: x_train, Y: y_train})
  #train_writer.add_summary(summary_tb, i)
 
    
#train_writer.close()

curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Loss: %s Output: %s "%(curr_loss,curr_Output))

curr_OutputN=np.zeros([126,3],dtype=np.float64)

for i in range(0,126):
    curr_OutputN[i,0] = ((y_maximo-y_minimo)*((curr_Output[i,0]+1)/2))+y_minimo
    curr_OutputN[i,1] = ((y_maximo-y_minimo)*((curr_Output[i,1]+1)/2))+y_minimo
    curr_OutputN[i,2] = ((y_maximo-y_minimo)*((curr_Output[i,2]+1)/2))+y_minimo

#Para la matriz de confusión se necesita la posición de la neurona que tuvo mayor #activación, esto determina la clase
for i in range(0,126):
    Out_CM[i] = np.argmax(curr_OutputN[i,:]) #Neurona que obtuvo la salida ganadora

#Nombres de las clases para la matriz de confusión    
class_names=['Class A', 'Class B', 'Class C']

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
data2=np.zeros([52,14],dtype=np.float64)
x_test=np.zeros([52,13],dtype=np.float64)
y_test=np.zeros([52,3],dtype=np.float64)
y_test_CM=np.zeros([52,1],dtype=np.float64)
Out_test_CM=np.zeros([52,1],dtype=np.float64)
#Lectura de los datos de entrenamiento desde el archivo
cont=0;

with open(DATA_FILE_TEST) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data2[cont,:]=(np.asarray(row))  
        x_test[cont,0] = data2[cont,1]
        x_test[cont,1] = data2[cont,2]        
        x_test[cont,2] = data2[cont,3]
        x_test[cont,3] = data2[cont,4]
        x_test[cont,4] = data2[cont,5]
        x_test[cont,5] = data2[cont,6]        
        x_test[cont,6] = data2[cont,7]
        x_test[cont,7] = data2[cont,8]
        x_test[cont,8] = data2[cont,9]
        x_test[cont,9] = data2[cont,10]        
        x_test[cont,10] = data2[cont,11]
        x_test[cont,11] = data2[cont,12]
        x_test[cont,12] = data2[cont,13]
        y_test_CM[cont,0] = data2[cont,0]
        #En archivo la clase viene como 0, 1 y 2. En este caso lo habitual es que quede una 
#neurona activa por clase. Es decir dependiendo de la clase solo se activará una #neurona
        if data2[cont,0]==1:
               y_test[cont,0] =1
               y_test[cont,1] =0
               y_test[cont,2] =0                     
        if data2[cont,0]==2:
               y_test[cont,0] =0
               y_test[cont,1] =1
               y_test[cont,2] =0 
        if data2[cont,0]==3:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =1                       
        cont=cont+1
n_samples=cont 
print(n_samples) 

x_testN=np.zeros([52,13],dtype=np.float64)
y_testN=np.zeros([52,3],dtype=np.float64)
for i in range(0,52):
    x_testN[i,0] = (2*((x_testN[i,0]-x1_minimo)/(x1_maximo-x1_minimo)))+(-1)
    x_testN[i,1] = (2*((x_testN[i,1]-x2_minimo)/(x2_maximo-x2_minimo)))+(-1)
    x_testN[i,2] = (2*((x_testN[i,2]-x3_minimo)/(x3_maximo-x3_minimo)))+(-1)
    x_testN[i,3] = (2*((x_testN[i,3]-x4_minimo)/(x4_maximo-x4_minimo)))+(-1)
    x_testN[i,4] = (2*((x_testN[i,4]-x5_minimo)/(x5_maximo-x5_minimo)))+(-1)
    x_testN[i,5] = (2*((x_testN[i,5]-x6_minimo)/(x6_maximo-x6_minimo)))+(-1)
    x_testN[i,6] = (2*((x_testN[i,6]-x7_minimo)/(x7_maximo-x7_minimo)))+(-1)
    x_testN[i,7] = (2*((x_testN[i,7]-x8_minimo)/(x8_maximo-x8_minimo)))+(-1)
    x_testN[i,8] = (2*((x_testN[i,8]-x9_minimo)/(x9_maximo-x9_minimo)))+(-1)
    x_testN[i,9] = (2*((x_testN[i,9]-x10_minimo)/(x10_maximo-x10_minimo)))+(-1)
    x_testN[i,10] = (2*((x_testN[i,10]-x11_minimo)/(x11_maximo-x11_minimo)))+(-1)
    x_testN[i,11] = (2*((x_testN[i,11]-x12_minimo)/(x12_maximo-x12_minimo)))+(-1)
    x_testN[i,12] = (2*((x_testN[i,12]-x13_minimo)/(x13_maximo-x13_minimo)))+(-1)
    y_testN[i,0] = (2*((y_testN[i,0]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    y_testN[i,1] = (2*((y_testN[i,1]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    y_testN[i,2] = (2*((y_testN[i,2]-y_minimo)/(y_maximo-y_minimo)))+(-1)
    
x_test=x_testN  
y_test=y_testN  

curr_loss2, curr_Output2 = sess.run([loss,Output], {X:  x_test, Y: y_test})
#curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("loss2: %s"%(curr_loss2))

curr_Output2N=np.zeros([52,3],dtype=np.float64)

for i in range(0,50):
    curr_Output2N[i,0] = ((y_maximo-y_minimo)*((curr_Output2[i,0]+1)/2))+y_minimo
    curr_Output2N[i,1] = ((y_maximo-y_minimo)*((curr_Output2[i,1]+1)/2))+y_minimo
    curr_Output2N[i,2] = ((y_maximo-y_minimo)*((curr_Output2[i,2]+1)/2))+y_minimo

for ii in range(0,52):
    Out_test_CM[ii] = np.argmax(curr_Output2N[ii,:])
       
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
