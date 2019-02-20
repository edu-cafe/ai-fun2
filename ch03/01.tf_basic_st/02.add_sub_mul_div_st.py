# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant(5, name='a')
b = tf.constant(2, name='b')

add1 = a + b
sub1 = a - b
mul1 = a * b
div1 = a / b
add2 = tf.____(a, b, name='add')
sub2 = tf._____(a, b, name='sub')
mul2 = tf.______(a, b, name='mul')
div2 = tf.____(a, b, name='div')
div3 = tf._______(a, b, name='divide')

sess = tf.Session()
print(sess.run(add1), sess.run(sub1), sess.run(mul1), sess.run(div1))
print(sess.run(add2), sess.run(sub2), sess.run(mul2), sess.run(div2), sess.run(div3))


