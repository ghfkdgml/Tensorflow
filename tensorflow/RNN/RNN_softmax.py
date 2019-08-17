import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dict = {c:i for i, c in enumerate(char_set)}

num_classes = len(char_set)
data_dim = len(char_set)
rnn_hidden_size = len(char_set)
sequence_length = 10 #any number
learning_rate = 0.1

dataX = []
dataY = []
for i in range(0,len(sentence) - sequence_length):
    x_str = sentence[i:i + sentence_length]
    y_str = sentence[i+1: i+ sequence_length + 1]
    x = [char_dict[c] for c in x_str]
    y = [char_dict[c] for c in y_str]
    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units= rnn_hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell * 2], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
