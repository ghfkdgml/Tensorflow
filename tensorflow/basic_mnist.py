import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#read mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#parameters
batch_xs, batch_ys = mnist.train.next_batch(100)
training_epochs = 15
batch_size = 100
nb_classes = 10

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, nb_classes])
W = tf.Variable(tf.random_normal([784,nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(x,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={x:batch_xs, y:batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' %(epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
    
    
