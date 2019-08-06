import tensorflow as tf
import numpy as np

xData = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
yData = np.array([[0],[1],[1],[0]], dtype=np.float32)

TrainNum=10000          #the number of training

def LogisticRegression():
    W1 = tf.Variable(tf.random_normal([2,2]),name='weight1')
    b1 = tf.Variable(tf.random_normal([2]),name='bias1')

    W2 = tf.Variable(tf.random_normal([2,1]),name='weight2')
    b2 = tf.Variable(tf.random_normal([1]),name='bias2')

    X = tf.placeholder(tf.float32,shape=[None,2])
    Y = tf.placeholder(tf.float32,shape=[None,1])

    #Hypothesis
    H1 = tf.sigmoid(tf.matmul(X,W1)+b1)
    H2 = tf.sigmoid(tf.matmul(H1,W2)+b2)
    #cost function
    cost = -tf.reduce_mean(Y * tf.log(H2) + (1 - Y) * tf.log(1 - H2))
    #train
    learning_rate = tf.Variable(0.01)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    train=optimizer.minimize(cost)

    predicted = tf.cast(H > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))
    #initialize variables
    init=tf.global_variables_initializer()
    #session
    sess=tf.Session()
    sess.run(init)
    for i in range(TrainNum):
        cost_val,_ = sess.run([cost, train], feed_dict={X:xData,Y:yData})
        if i%1000 ==0:
            print(i,sess.run(cost,feed_dict={X:xData,Y:yData}),sess.run(W),sess.run(b))
    h,c,a = sess.run([H, predicted, accuracy], feed_dict={X:xData,Y:yData})

    #Prediction Result
    print(h,c,a)


if __name__=='__main__':
    LogisticRegression()
