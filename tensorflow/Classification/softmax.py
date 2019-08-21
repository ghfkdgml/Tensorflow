import tensorflow as tf
import numpy as np

xy = np.loadtxt('data4_zoo.csv', delimiter=',', dtype=np.float32)
xData = xy[:,:-1]
yData = xy[:,[-1]]

TrainNum = 100         #the number of training
nb_classes = 7
def LogisticRegression():
    W = tf.Variable(tf.random_normal([16,nb_classes]),name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]),name='bias')
    X = tf.placeholder(tf.float32,shape=[None,16])
    Y = tf.placeholder(tf.int32,shape=[None,1])
    Y_one_hot = tf.one_hot(Y, nb_classes)           #[[[1,2,3],[4,5,6]]]
    Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes])#[[[1,2,3],[4,5,6]]] --> [[1,2,3],[4,5,6]]

    #Hypothesis
    logits = tf.matmul(X,W)+b
    H = tf.nn.softmax(logits)
    #cost function

    #cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(H),axis=1))
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

    #train
    learning_rate = tf.Variable(0.01)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    cost = tf.reduce_mean(cost_i)
    train = optimizer.minimize(cost)

    prediction = tf.argmax(H,1)
    correct_prediction = tf.equal(prediction,tf.argmax(Y_one_hot,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #initialize variables
    init = tf.global_variables_initializer()
    #session
    sess = tf.Session()
    sess.run(init)
    for i in range(TrainNum):
        cost_val,_ = sess.run([cost, train], feed_dict={X:xData,Y:yData})
        if i % 1000 == 0:
            print(i,sess.run(cost,feed_dict={X:xData,Y:yData}),sess.run(W),sess.run(b))
    pred = sess.run(prediction, feed_dict={X:xData})
    for p, y in zip(pred, yData.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y),p,int(y)))

    #Prediction Result
    a = sess.run(H,feed_dict={X:[[0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]]})
    #print(sess.run(H,feed_dict={X:[1,11,7,9]}))
    print(sess.run(tf.arg_max(a,1)))


if __name__=='__main__':
    LogisticRegression()
