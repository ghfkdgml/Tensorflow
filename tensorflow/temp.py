import tensorflow as tf

a = tf.constant(20, name="a")
b = tf.constant(30, name="b")
mul = a * b

sess = tf.Session()
tw = tf.train.SummaryWriter("log_dir", graph=sess.graph)

print(sess.run(mul))
