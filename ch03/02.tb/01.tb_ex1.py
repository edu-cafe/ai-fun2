# tensorboard ex1
import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

#writer = tf.summary.FileWriter('c:/share/tb/ex1', sess.graph)  #ok
writer = tf.summary.FileWriter('/share/tb/tb1', sess.graph)  #ok
sess.run(x)
writer.close()
sess.close()
