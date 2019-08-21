import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#read mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#parameters
batch_xs, batch_ys = mnist.train.next_batch(100)
training_epochs = 15
batch_size = 100
nb_classes = 512
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784,nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([nb_classes]))
L1 = tf.nn.relu(tf.matmul(x,W1) + b1)      #sigmoid -> relu ft
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)#choose some x (not all x)

W2 = tf.get_variable("W2", shape=[nb_classes,nb_classes], initializer=tf.contrib.layers.xavier_initializer()) #xavier is used to make initializer more efficient
b2 = tf.Variable(tf.random_normal([nb_classes]))
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


W3 = tf.get_variable("W3", shape=[nb_classes, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([nb_classes]))
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)


W4 = tf.get_variable("W4", shape=[nb_classes, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]))
L4 = tf.nn.relu(tf.matmul(L3,W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[nb_classes, 10], initializer=tf.contrib.layers.xavier_initializer())# 0~9 -> 10
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4,W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})#train : keep_prob: 0.5 ~ 0.7
            avg_cost += c / total_batch

        print('Epoch:', '%04d' %(epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Accuracy:', sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1})) #test : keep_prob: 1.0
    
    
    
