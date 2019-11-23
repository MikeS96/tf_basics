# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:53:40 2018

@author: Cortana
"""

import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b

print(a)
print(b)
print(a+b)

writer = tf.summary.FileWriter('.') #Esto es para generar le grafo
writer.add_graph(tf.get_default_graph())

sess = tf.Session() #Inicia la sesión en Tensorflow para poder imprimir
print(sess.run(total))
print(sess.run(a))
print(sess.run(b))
