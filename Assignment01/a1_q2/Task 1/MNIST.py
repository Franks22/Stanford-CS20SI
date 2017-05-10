"""
Starter code for logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 100
n_hidden_1 = 256 # Number of hidden neurons in first layer
n_hidden_2 = 256 # Number of hidden neurons in first layer
size_input = 784 # Input size of MNIST data (img shape 28*28)
size_output = 10 # total number of classes (from 0 to 9)

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
X = tf.placeholder(tf.float32, [batch_size, size_input], name='X_image')
Y = tf.placeholder(tf.float32, [batch_size, size_output], name='Y_label')

# Step 3: create weights and bias
# weights are initialized random
W1 = tf.Variable(tf.truncated_normal([size_input, n_hidden_1], stddev = 0.1), name='weights1')
W2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev = 0.1), name='weights2')
W3 = tf.Variable(tf.truncated_normal([n_hidden_2, size_output], stddev = 0.1), name='weights3')
# Biases
#b1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev = 0.1, mean = 0.3), name='bias1')
#b2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev = 0.1, mean = 0.3), name='bias2')
#b3 = tf.Variable(tf.truncated_normal([size_output], stddev = 0.1, mean = 0.3), name='bias3')
b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_1]), name='bias1')
b2 = tf.Variable(tf.constant(0.1, shape=[n_hidden_2]), name='bias2')
b3 = tf.Variable(tf.constant(0.1, shape=[size_output]), name='bias3')


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
layer_1 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(X,W1), b1)))
layer_2 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1,W2), b2)))
logits = tf.add(tf.matmul(layer_2,W3), b3)


# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits)
loss = tf.reduce_mean(entropy)

# Step 6: define training op
# using gradient descent to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./my_graphs/logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			# Run optimizer + fetch loss_batch
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
			total_loss += loss_batch
		print 'Average loss epoch {0}: {1}'.format(i, total_loss/n_batches)

	print 'Total time: {0} seconds'.format(time.time() - start_time)

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch}) 
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds += sess.run(accuracy)	
	
	print 'Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples)
