""" Predicting coronary heart disease
"""

import csv
import numpy as np
import numpy.core.defchararray as np_f
import tensorflow as tf

PATH_TO_DATA = 'heart.csv'
NUM_OF_SAMPLES = 462
NUM_OF_TEST_SAMPLES = 92 # roughly 20% of the whole data
NUM_OF_FEATURES = 9
NUM_OF_CLASSES = 2
LEARNING_RATE = 0.000001
NUM_EPOCHS = 500000
SKIP_STEP = 2000

def _read_data():
    
    all_data=[]

    file = open(PATH_TO_DATA)
    csv_file = csv.reader(file)

    for line in csv_file:
        all_data.append(line)

    all_data.pop(0) # Delete the first Element of the list (headers)

    np.random.shuffle(all_data) # Random shuffle the elements

    Data = np.asarray(all_data) # Convert to Array
    # Convert "Absent" and "Present" to 0 and 1se
    Data = np_f.replace(Data, 'Absent', '0.0')
    Data = np_f.replace(Data, 'Present', '1.0')
    
    X_str,Y_str = np.hsplit(Data,np.array([9]))

    X = X_str.astype(np.float)
    Y_temp = Y_str.astype(np.int)

    # Convert Y into a one-hot representation
    Y = np.concatenate((np.abs(Y_temp-1),Y_temp),axis=1)

    return X,Y

def _split_data(X, Y, no_test_samples):
    X_test = X[:NUM_OF_TEST_SAMPLES,:]
    X_train = X[NUM_OF_TEST_SAMPLES:,:]
    Y_test = Y[:NUM_OF_TEST_SAMPLES]
    Y_train = Y[NUM_OF_TEST_SAMPLES:]
    
    return X_train,Y_train,X_test,Y_test

class LogisticClassifier:
    def __init__(self, learning_rate, batch_size, num_features, num_classes):
        self.batch_size = batch_size
        self.features = num_features
        self.classes = num_classes
        self.lr = learning_rate
        self.global_step = tf.Variable(0,dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.X = tf.placeholder(tf.float32, shape=[None,self.features], name='Features')
            self.Y = tf.placeholder(tf.int32, shape=[None,self.classes], name='Labels')

    def _create_weights(self):
        self.weights = tf.Variable(tf.zeros([self.features, self.classes]), name='weights')
        self.b = tf.Variable(tf.zeros([self.classes]), name='biases')

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.logits = tf.matmul(self.X,self.weights) + self.b
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits, name='entropy')
            self.loss = tf.reduce_mean(entropy, name='loss')

    def _create_optimizer(self):
        self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_weights()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

def train_model(model, X,Y):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('my_graphs/a1_q2', sess.graph)

        for i in range(NUM_EPOCHS):
            total_loss = 0.0

            total_loss,_,summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict={model.X: X, model.Y: Y})

            writer.add_summary(summary, global_step=i)
            if (i+1) % SKIP_STEP == 0:
                print 'Loss Epoch {0}: {1}'.format(i, total_loss)

def test_model(model, X_test, Y_test):
	# test the model
	total_correct_preds = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		_, loss_test, logits_test = sess.run([model.optimizer, model.loss, model.logits], feed_dict={model.X: X_test, model.Y:Y_test}) 
		preds = tf.nn.softmax(logits_test)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_test, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds = sess.run(accuracy)
		acc = total_correct_preds/NUM_OF_TEST_SAMPLES
		print acc


def main():
    X,Y = _read_data()
    X_train,Y_train,X_test,Y_test = _split_data(X, Y, NUM_OF_TEST_SAMPLES)
    model = LogisticClassifier(LEARNING_RATE, NUM_OF_SAMPLES - NUM_OF_TEST_SAMPLES, NUM_OF_FEATURES, NUM_OF_CLASSES)
    model.build_graph()
    train_model(model, X_train, Y_train)
    test_model(model, X_test, Y_test)

if __name__ == '__main__':
    main()