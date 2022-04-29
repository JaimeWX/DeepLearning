import tensorflow as tf
import numpy as np

A = tf.constant([[1,2,3],[4,5,6]])
print(A.dtype)
AA = tf.cast(A,tf.float64)
Y = tf.constant([[0,0,1],[0,1,0]])
YY = tf.cast(Y,tf.float64)
loss = tf.keras.losses.mse(YY, AA)
print(loss)