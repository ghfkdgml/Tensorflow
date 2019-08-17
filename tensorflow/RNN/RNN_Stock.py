import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


#Parameters
seq_len = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

xy_data = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy_data[::-1] #reverse order

train_size = int(len(xy) * 0.7)
train_set = xy[:train_size]
test_set = xy[train_size - seq_len:]

train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # we need only last value as result
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_len)
testX, testY = build_dataset(test_set, seq_len)

X = tf.placeholder(tf.float32, [None, seq_len,data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.BasicLSTMCell(num_units= hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)

loss = tf.reduce_mean(tf.square(Y_pred - Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: trainX, Y: trainY})
        print(i, "loss:",l)
    result = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: result})
    print("RMSE: {}".format(rmse_val))
    print("pred_value: {}".format(result))
