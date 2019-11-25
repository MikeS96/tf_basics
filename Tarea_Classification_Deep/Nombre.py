# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:02:54 2018

@author: Cortana
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import itertools
from sklearn.metrics import confusion_matrix

#Definición de los arreglos que almacenarán los datos de entrenamiento
x_train=[[1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1],[1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1],[1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1],[1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1],[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1]]
y_train=[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
y_train_CM=[[0],[1],[2],[3],[4]]
Out_CM=np.zeros([5,1],dtype=np.float64)

# Definición de los place holders para los datos de entrenamiento y validación
# como se tiene una cantidad de datos diferentes para entrenar (120)  y para validar #(30)  se deja indefinido el tamaño 
X = tf.placeholder(tf.float32,(None, None), name='X')
Y = tf.placeholder(tf.float32,(None, None), name='Y')



# Definición de las variables para los pesos de la RNA
Wco = tf.Variable(tf.random_uniform((35,20),-1,1))
bco = tf.Variable(tf.random_uniform((20,),-1,1))

Wcs = tf.Variable(tf.random_uniform((20,5),-1,1))
bcs = tf.Variable(tf.random_uniform((5,),-1,1))

#Calculo de la salida.
#Capa oculta tangente sigmoidal
#Capa de salida sigmoidal
OutputCo= tf.tanh(tf.matmul(X,Wco)+bco)
Output=tf.nn.sigmoid((tf.matmul(OutputCo,Wcs)+bcs))

# Función de pérdida MSE
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizador
optimizer  = tf.train.AdagradOptimizer(0.1,0.0001)

train = optimizer.minimize(loss)
# Entrenamiento de la red
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init) 

# Se entrena la red 5000 iteracioens
for i in range(20000):
  sess.run(train, {X: x_train, Y: y_train})

# Se prueba la red con los datos de entrenamiento
curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Loss: %s Output: %s "%(curr_loss,curr_Output))

#Cosa mia para poder imprimir la imagen y visualizar la entrada
a = np.asarray(x_train)
x = np.zeros([1,35],dtype=np.float64)
u = np.zeros([7,5],dtype=np.float64)

for j in range(0,5):
  x = a[j,:]
  u = np.reshape(x, (7,5))
  plt.imshow(u, interpolation='nearest')
  plt.show()

#Para la matriz de confusión se necesita la posición de la neurona que tuvo mayor #activación, esto determina la clase
for i in range(0,5):
    Out_CM[i] = np.argmax(curr_Output[i,:]) #Neurona que obtuvo la salida ganadora

#Nombres de las clases para la matriz de confusión    
class_names=['A', 'N', 'G', 'E', 'L']

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
x_test=[[1,1,1,1,0,1,0,1,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,0],[1,0,0,0,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,0,0,0,1],[1,1,1,1,1,1,0,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,0],[1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1],[1,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,1,1]]
y_test=[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
y_test_CM=[[0],[1],[2],[3],[4]]
Out_test_CM=np.zeros([5,1],dtype=np.float64)
#Lectura de los datos de entrenamiento desde el archivo

# Se prueba la red con los datos de validación
curr_loss, curr_Output = sess.run([loss,Output], {X:  x_test, Y: y_test})
print("Loss: %s Output: %s "%(curr_loss,curr_Output))


a = np.asarray(x_test)

for j in range(0,5):
  x = a[j,:]
  u = np.reshape(x, (7,5))
  plt.imshow(u, interpolation='nearest')
  plt.show()



for ii in range(0,5):
    Out_test_CM[ii] = np.argmax(curr_Output[ii,:])
       
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
