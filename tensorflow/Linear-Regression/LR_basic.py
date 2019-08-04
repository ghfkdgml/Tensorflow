import tensorflow as tf

xData=[1,2,3,4,5,6,7]
yData=[25000,55000,75000,110000,128000,155000,180000]
predictData=[8]        #data list that you want to predict

TrainNum=5000          #the number of training

def LinearRegression():
    W=tf.Variable(tf.random_uniform([1],-100,100))
    b=tf.Variable(tf.random_uniform([1],-100,100))
    X=tf.placeholder(tf.float32)
    Y=tf.placeholder(tf.float32)

    #Hypothesis
    H=W*X+b
    #cost function
    cost=tf.reduce_mean(tf.square(H-Y))
    #train
    a=tf.Variable(0.01)
    optimizer=tf.train.GradientDescentOptimizer(a)
    train=optimizer.minimize(cost)
    #initialize variables
    init=tf.global_variables_initializer()
    #session
    sess=tf.Session()
    sess.run(init)
    for i in range(TrainNum):
        sess.run(train, feed_dict={X:xData,Y:yData})
        if i%500 ==0:
            print(i,sess.run(cost,feed_dict={X:xData,Y:yData}),sess.run(W),sess.run(b))

    #Prediction Result
    print(sess.run(H,feed_dict={X:predictData}))


if __name__=='__main__':
    LinearRegression()
