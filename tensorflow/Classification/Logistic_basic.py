import tensorflow as tf

xData=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
yData=[[0],[0],[0],[1],[1],[1]]

TrainNum=10000          #the number of training

def LogisticRegression():
    W=tf.Variable(tf.random_normal([2,1]),name='weight')
    b=tf.Variable(tf.random_normal([1]),name='bias')
    X=tf.placeholder(tf.float32,shape=[None,2])
    Y=tf.placeholder(tf.float32,shape=[None,1])

    #Hypothesis
    H=tf.sigmoid(tf.matmul(X,W)+b)
    #cost function
    cost = -tf.reduce_mean(Y * tf.log(H) + (1 - Y) * tf.log(1 - H))
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
