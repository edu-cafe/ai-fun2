# tensorboard  ex5
import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

add = tf.add(X, Y)
mul = tf.multiply(X, Y)

# step 1: node 선택
add_hist = tf.summary.scalar('add_scalar', add)
mul_hist = tf.summary.scalar('mul_scalar', mul)

# step 2: summary 통합. 두 개의 코드 모두 동작.
merged = tf.summary.merge_all()
# merged = tf.summary.merge([add_hist, mul_hist])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # step 3: writer 생성
    writer = tf.summary.FileWriter('/share/tb/tb5', sess.graph)

    for step in range(100):
        # step 4: 노드 추가
        summary = sess.run(merged, feed_dict={X: step * 1.0, Y: 2.0})
        writer.add_summary(summary, step)

