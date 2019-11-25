# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 02:04:38 2018

@author: Cortana
"""

import tensorflow as tf

x=tf.placeholder(tf.float32)
w=tf.Variable([[0.093],[0.084]],dtype=tf.float32) #Los pesos estáran cambiando cuando la red se vaya a entrenar
b=tf.constant([-0.018],dtype=tf.float32)

Neta=tf.matmul(x,w)+b
Salida=tf.nn.relu(Neta)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

ValorNeta=sess.run(Neta,{x:[[0,0],[0,1],[1,0],[1,1]]})
ValorSalida=sess.run(Salida,{x:[[0,0],[0,1],[1,0],[1,1]]})


print("Neta: %s"%(ValorNeta))
print("Salida: %s"%(ValorSalida))