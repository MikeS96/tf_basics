# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 01:55:21 2018

@author: Cortana
"""

import tensorflow as tf

x=tf.constant(-3.0)
w=tf.constant(3.0)
b=tf.constant(-1.5)

SalidaAux=tf.multiply(x,w)
Neta=tf.add(SalidaAux,b)
Salida=tf.nn.relu(Neta)

sess=tf.Session()
ValorSalida=sess.run(Salida) #Corro los valores que quiero ver
ValorNeta=sess.run(Neta)

print("Neta", ValorNeta)
print("Salida: %s"%ValorSalida)