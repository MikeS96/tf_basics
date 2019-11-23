import numpy as np
import matplotlib.pyplot as plt
import math

x_train=np.zeros([21,1],dtype=np.float64)
y_train=np.zeros([21,1],dtype=np.float64)
cont=0;

for i in range(0,21):
    x_train[i] = (math.pi)*i*0.1
    y_train[i] = math.sin(x_train[i] )
   
Xg, Yg = x_train, y_train
plt.plot(Xg, Yg, 'bo', label='Datos Deseados')
plt.legend()
plt.show()

