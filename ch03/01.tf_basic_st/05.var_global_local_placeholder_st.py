# https://www.tensorflow.org/api_docs/python/tf/

import tensorflow as tf

vl = tf._____________()
vl = [[1,10],20,30]
print(vl)

vg = tf.______(tf.zeros(3, dtype=tf.int32), name='vg')
sess = tf.Session()
sess.run(tf.______________())
print(sess.run(vg))

td = tf.zeros( .... )
print(sess.run(td))    # [[0. 0.][0. 0.][0. 0.]]

ta = tf.placeholder(tf.float32, (2,2))
tb = tf.placeholder(tf.float32, (1,2))
tc = tf.multiply(ta, tb)
print(sess.run(tc, feed_dict={ ......... }))

sess.close()