import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm, metrics

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

# initialize
batch_size = 1000
# train model
print('Learning started. It takes sometime.')
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print(batch_xs[0])
#clf = svm.SVC()
#clf.fit(batch_xs,batch_ys)
#test = mnist.test.images[:300]
#predict = clf.predict(test)
## Get one and predict
#score = metrics.accuracy_score(mnist.test.labels[:300],predict)
#print(score)
