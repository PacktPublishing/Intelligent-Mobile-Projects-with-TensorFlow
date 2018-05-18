import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import *
import matplotlib.pyplot as plt

num_neurons = 100
num_inputs = 1
num_outputs = 1
symbol = 'goog'
epochs = 500
seq_len = 20
learning_rate = 0.001



f = open(symbol + '.txt', 'r').read()
data = f.split('\n')[:-1] # get rid of the last '' so float(n) works
data.reverse()
d = [float(n) for n in data]

result = []
for i in range(len(d) - seq_len - 1):
    result.append(d[i: i + seq_len + 1])

result = np.array(result)

row = int(round(0.9 * result.shape[0]))
train = result[:row, :] 
test = result[row:, :]

# normally  you should most likely randomly shuffle your data 
# before splitting it into a training and test set.
np.random.shuffle(train) 

X_train = train[:, :-1] # all rows with all columns except the last one
X_test = test[:, :-1] # rest 20% used for testing

y_train = train[:, 1:] 
y_test = test[:, 1:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_inputs))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_inputs))  
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], num_outputs))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], num_outputs))  

X = tf.placeholder(tf.float32, [None, seq_len, num_inputs])
y = tf.placeholder(tf.float32, [None, seq_len, num_outputs])

# At each time step we have an output vector of size 100 (num_neurons). But what we actually want 
# is a single output value at each time step. The simplest solution is to wrap the cell in 
# an OutputProjectionWrapper, which adds a fully connected layer of linear num_neurons (i.e., 
# without any activation function) on top of each output (but it does not affect the cell state).
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu), 
    # BasicLSTMCell BasicRNNCell
    output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

preds = tf.reshape(outputs, [1, seq_len], name="preds")
loss = tf.reduce_mean(tf.square(outputs - y)) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()

    count = 0
    for _ in range(epochs):
        n=0
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        count += 1
        if count % 10 == 0: # save checkpoint every 10 iterations
            saver.save(sess, "model/" + symbol + "_model.ckpt") 
            loss_val = loss.eval(feed_dict={X: X_train, y: y_train})
            print(count, "loss:", loss_val)

    correct = 0
    y_pred = sess.run(outputs, feed_dict={X: X_test})    
    targets = []
    predictions = []
    for i in range(y_pred.shape[0]):
        input = X_test[i]
        target = y_test[i]
        prediction = y_pred[i]

        targets.append(target[-1][0])
        predictions.append(prediction[-1][0])

        if target[-1][0] >= input[-1][0] and prediction[-1][0] >= input[-1][0]:
            correct += 1
        elif target[-1][0] < input[-1][0] and prediction[-1][0] < input[-1][0]:
            correct += 1

    total = len(X_test)
    xs = [i for i, _ in enumerate(y_test)]
    plt.plot(xs, predictions, 'r-', label='prediction') # red dot-dashed line: 'r-.'
    plt.plot(xs, targets, 'b-', label='true') # green solid line 
    plt.legend(loc=0)
    plt.title("%s - %d/%d=%.2f%%" %(symbol, correct, total, 100*float(correct)/total))
    plt.show()

