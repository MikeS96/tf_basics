# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:23 2018

@author: Cortana
"""

import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

sess=tf.Session()

print(sess.run(z, feed_dict={x: 3, y: 4.5}))#Evaluo modelo para ver Z, y  con feed_dic asigno los valores a los place holder
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
#Feed_dict se encarga de asignar valores al place holder