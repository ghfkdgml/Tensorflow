import tensorflow as tf
import numpy as np

xy = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)#load data from csv file
xData = xy[:,0:-1]
yData = xy[:,[-1]]

#xData=[[73.,80.,75.],[93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
#yData=[[152.],[185.],[180.],[196.],[142.]]
#predictData=[8]        #data list that you want to predict

TrainNum=2000          #the number of training

def LinearRegression():
    W=tf.Variable(tf.random_normal([3,1],name = 'weight'))
    b=tf.Variable(tf.random_normal([1],name = 'bias'))
    X=tf.placeholder(tf.float32,shape = [None,3])
    Y=tf.placeholder(tf.float32,shape = [None,1])

    #Hypothesis
    H=tf.matmul(X,W)+b
    #cost function
    cost=tf.reduce_mean(tf.square(H-Y))
    #train
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train=optimizer.minimize(cost)
    #initialize variables
    init=tf.global_variables_initializer()
    #session
    sess=tf.Session()
    sess.run(init)
    for i in range(TrainNum):
        result,_ = sess.run([H,train], feed_dict={X:xData,Y:yData})
        if i%500 ==0:
            print(result)
    print(sess.run([H],feed_dict={X:[[100,70,101]]}))

    #Prediction Result


if __name__=='__main__':
    LinearRegression()
