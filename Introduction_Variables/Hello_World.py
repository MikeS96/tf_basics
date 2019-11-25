# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:51:10 2018

@author: Cortana
"""

import tensorflow as tf
hello=tf.constant('Hello,Tensorflow')
sess=tf.Session()
print(sess.run(hello))