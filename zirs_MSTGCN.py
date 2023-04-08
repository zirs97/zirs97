import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import models
from keras.layers import Layer
from keras.layers.core import Dropout, Lambda
from tensorflow.python.framework import ops

x = tf.random.normal(shape=(32, 8064, 12, 5))
    # Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features) #(32, 8064, 12,5)
    # Output: (batch_size, num_of_vertices, num_of_vertices)
a = tf.random.normal(shape=(5,1)) #shape=(num_of_features, 1),
_, T, V, F = x.shape
N = tf.shape(x)[0]

outputs = []
diff_tmp = 0
for time_step in range(T):
    # shape: (N,V,F) use the current slice
    xt = x[:, time_step, :, :] #(N,V,F)
    x1 = tf.transpose(tf.broadcast_to(xt, [V,N,V,F]), perm=[2,1,0,3]) ## shape: (V,N,V,F)
    # (N,V,F) - (V,N,V,F) =(V,N,V,F)  再转置操作 (V, N,V,F)  两个节点交换位置
    diff = tf.transpose(tf.broadcast_to(xt, [V,N,V,F]), perm=[2,1,0,3]) - xt      #F那一维度去哪了 ##(V,N,V,F)
    # shape: (N,V,V)
    #求绝对值之后transpose(N,V,V,F) 再点乘a后为(N,V,V,1) 再reshape到(N,V,V)
    tmpS = K.exp(K.reshape(K.dot(tf.transpose(K.abs(diff), perm=[1,0,2,3]), a), [N,V,V])) #(32, 12, 12)
    # normalization
    S = tmpS / tf.transpose(tf.broadcast_to(K.sum(tmpS, axis=1), [V,N,V]), perm=[1,2,0])

    diff_tmp += K.abs(diff)
    outputs.append(S)

outputs = tf.transpose(outputs, perm=[1,0,2,3])
S = K.mean(outputs, axis=0)
diff = K.mean(diff_tmp, axis=0) /tf.convert_to_tensor(int(T), tf.float32)
print(outputs)

