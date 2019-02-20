# tensorboard ex2
import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b)

writer = tf.summary.FileWriter('/share/tb/tb2', sess.graph)
sess.run(x)
writer.close()
sess.close()
