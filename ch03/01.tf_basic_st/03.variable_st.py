# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant(5, name='a')
b = tf.constant(2, name='b')

va = tf._______(5, name='va')
vb = tf._______(3, name='vb')
vc = tf._______(tf.zeros(0, tf.int32), name='vc')
#print(va)
#print(vb)
#print(vc)

sess = tf.Session()

print(sess.run(a))    #ok
#print(sess.run(va))      #error

sess.run(tf._________________)
print(sess.run(va))

print('-----------------')
#Tensor.eval returns a numpy array with the same contents as the tensor.
print(va.____(sess))
print(vb.____(sess))
print(vc.____(sess))

print('-----------------')

va = va + 10
vb = vb - 5
vc = va
vc = vc + vb
print(sess.run(va))
print(sess.run(vb))
print(sess.run(vc))






