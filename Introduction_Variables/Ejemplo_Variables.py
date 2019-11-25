# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:59:34 2018

@author: Cortana
"""

import tensorflow as tf

x = tf.constant(35, name='x')
y = tf.Variable(x+5, name='y') #Creo una variable que es X + 5 y 
#Especifico el nombre de la variable



init = tf.global_variables_initializer() #inicializo las variables
sess = tf.Session() #Crea una sesión

sess.run(init)#Ejecuto la sesión con todas las variables
print("x",sess.run(x)) #Imprimo X, valor de X
print("y",sess.run(y))
