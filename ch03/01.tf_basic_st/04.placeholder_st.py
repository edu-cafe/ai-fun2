# -*- coding: utf-8 -*-
import tensorflow as tf

va = tf.Variable(5.0, name='va')
pa = tf._________(tf.float32, name='pa')
#print(pa)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(va.eval(sess))
#print(pa.eval(sess))    #error
t = pa + 1.0
print(t.eval(session=sess, _________ ))

print('-----------------')

ta = tf.placeholder(tf.float32, 3)
tb = tf.placeholder(tf.float32, 1)
tc = tf.multiply(ta, tb)
print(sess.run(tc, ______________ ))


print('-----------------')

