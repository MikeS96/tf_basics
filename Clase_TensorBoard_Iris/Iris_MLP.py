
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import itertools


# Ruta de los archivos con los datos de entrenamiento y validación
DATA_FILE_TRAINING = 'data/iris_training_No_header.csv'
DATA_FILE_TEST = 'data/iris_test_No_header.csv' 
#DATA_FILE_PESOS = 'data/MLP_Pesos.csv'
      
#Definición de los arreglos que almacenarán los datos de entrenamiento
data=np.zeros([120,5],dtype=np.float64)
x_train=np.zeros([120,4],dtype=np.float64)
y_train=np.zeros([120,3],dtype=np.float64)
y_train_CM=np.zeros([120,1],dtype=np.float64)
Out_CM=np.zeros([120,1],dtype=np.float64)

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
        y_train_CM[cont]=data[cont,4]
#En archivo la clase viene como 0, 1 y 2. En este caso lo habitual es que quede una 
#neurona activa por clase. Es decir dependiendo de la clase solo se activará una #neurona
        if data[cont,4]==0:
               y_train[cont,0] =1
               y_train[cont,1] =0
               y_train[cont,2] =0                     
        if data[cont,4]==1:
               y_train[cont,0] =0
               y_train[cont,1] =1
               y_train[cont,2] =0 
        if data[cont,4]==2:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =1                       
        cont=cont+1

n_samples=cont 
print(n_samples)   

# Definición de los place holders para los datos de entrenamiento y validación
# como se tiene una cantidad de datos diferentes para entrenar (120)  y para validar #(30)  se deja indefinido el tamaño 
X = tf.placeholder(tf.float32,(None, None), name='X')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

# Definición de las variables para los pesos de la RNA
Wco = tf.Variable(tf.random_uniform((4,10),-1,1))
bco = tf.Variable(tf.random_uniform((10,),-1,1))

Wcs = tf.Variable(tf.random_uniform((10,3),-1,1))
bcs = tf.Variable(tf.random_uniform((3,),-1,1))

#Calculo de la salida.
#Capa oculta tangente sigmoidal
#Capa de salida sigmoidal
OutputCo= tf.tanh(tf.matmul(X,Wco)+bco)
Output=tf.nn.sigmoid((tf.matmul(OutputCo,Wcs)+bcs))

# Función de pérdida MSE
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizador
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
# Entrenamiento de la red
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init) 

# Se entrena la red 5000 iteracioens
for i in range(5000):
  sess.run(train, {X: x_train, Y: y_train})

# Se prueba la red con los datos de entrenamiento
curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Loss: %s Output: %s "%(curr_loss,curr_Output))