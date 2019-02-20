# TensorFlow Linear Regression Example
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt 
from sklearn import datasets

boston = datasets.load_boston()
boston_slice = [x[5] for x in boston.data]  #6번째 피처만 사용

data_x = np.array(boston_slice).reshape(-1,1)  
data_y = boston.target.reshape(-1,1) 

n_sample = data_x.shape[0]  # (506,1) -> 506
x = tf.placeholder(tf.float32, shape=(n_sample, 1), name='x')  #feature
y = tf.placeholder(tf.float32, shape=(n_sample, 1), name='y')  #target
w = tf.Variable(tf.zeros((1,1)), name='weights') 
b = tf.Variable(tf.zeros((1,1)), name='bias')

y_pred = tf.matmul(x,w) + b  #모델

loss = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y)))  #손실함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)  #최적화클래스(함수)
train_op = optimizer.minimize(loss)  #최적화함수, 손실함수의 최소값 찾기
summary_op = tf.summary.scalar('loss', loss)  #시각화를위한 서머리함수, 손실함수의 변화를 기록함

def plot_graph(y, fout):
    '''데이터 플롯을 위한 함수:입력값(피처값), 출력값(집값)'''
    plt.scatter(data_x.reshape(1, -1)[0], boston.target.reshape(1, -1)[0])
    plt.plot(data_x.reshape(1, -1)[0], y.reshape(1, -1)[0])
    plt.savefig(fout)
    plt.clf()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('/share/tb/ex7', sess.graph)
    y_pred_before = sess.run(y_pred, {x: data_x})
    plot_graph(y_pred_before, '/share/ai/ch03/03.선형회귀/before.png')

    # 최적화함수를 이용하여 기울기를 8000번 업데이트
    for i in range(8000):
        loss_rst, summary_rst, _ = sess.run([loss, summary_op, train_op], feed_dict={x: data_x, y: data_y})
        summary_writer.add_summary(summary_rst, i)
        if i % 100 == 0:
            print('loss : %4.4f' % loss_rst.mean())
            y_pred_after = sess.run(y_pred, {x: data_x})

    y_pred_after = sess.run(y_pred, {x: data_x})
    plot_graph(y_pred_after, '/share/ai/ch03/03.선형회귀/after.png')
    
    print('Predict Score => ')
    test_x = [6.575, 6.421, 7.001]
    test_y = test_x * w + b
    print(sess.run(test_y))


