import tensorflow as tf

xData = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
yData = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

TrainNum = 10000         #the number of training

def LogisticRegression():
    W = tf.Variable(tf.random_normal([4,3]),name='weight')
    b = tf.Variable(tf.random_normal([3]),name='bias')
    X = tf.placeholder(tf.float32,shape=[None,4])
    Y = tf.placeholder(tf.float32,shape=[None,3])
    Y_one_shot = tf.one_hot(Y, 3)
    Y_one_shot = tf.reshape(Y_one_hot,[-1,3])

    #Hypothesis
    logits = tf.matmul(X,W)+b
    H = tf.nn.softmax(logits)
    #cost function

    #cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(H),axis=1))
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

    #train
    learning_rate = tf.Variable(0.01)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    #predicted = tf.cast(H > 0.5, dtype=tf.float32)
    #accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))
    #initialize variables
    init = tf.global_variables_initializer()
    #session
    sess = tf.Session()
    sess.run(init)
    for i in range(TrainNum):
        cost_val,_ = sess.run([cost, train], feed_dict={X:xData,Y:yData})
        if i % 1000 == 0:
            print(i,sess.run(cost,feed_dict={X:xData,Y:yData}),sess.run(W),sess.run(b))
    #h,c,a = sess.run([H, predicted, accuracy], feed_dict={X:xData,Y:yData})

    #Prediction Result
    a = sess.run(H,feed_dict={X:[[1,11,7,9]]})
    #print(sess.run(H,feed_dict={X:[1,11,7,9]}))
    print(sess.run(tf.arg_max(a,1)))


if __name__=='__main__':
    LogisticRegression()
