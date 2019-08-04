import tensorflow as tf
import numpy as np

#get data from many files and use batch
filename_queue = tf.train.string_input_producer(
    ['test1.csv','test2.csv'],shuffle=False,
    name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]#define decode data type
xy = tf.decode_csv(value, record_defaults=record_defaults)

TrainNum=2000          #the number of training
train_x_batch,train_y_batch = tf.train.batch([xy[0:-1],xy[-1:]], batch_size=10)

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
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(TrainNum):
        x_batch, y_batch = sess.run([train_x_batch,train_y_batch])
        cost_result,result,_ = sess.run([cost,H,train], feed_dict={X:x_batch,Y:y_batch})
        if i%500 ==0:
            print("cost:",cost," result:",result)
    
    coord.request_stop()
    coord.join(threads)

if __name__=='__main__':
    LinearRegression()
