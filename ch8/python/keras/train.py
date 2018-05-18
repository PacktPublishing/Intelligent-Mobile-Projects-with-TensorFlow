import keras
from keras import backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

symbol = 'goog'
epochs = 10
num_neurons = 100
seq_len = 20 # aka window size
pred_len = 1 # number of days to predict prices with

shift_pred = False # predict xi+1, xi+2, xi+2, ..., xi+n based on input xi, xi+1, xi+2, ..., xi+n-1
# False means predict xi+n based on input xi, xi+1, xi+2, ..., xi+n-1


def load_data(filename, seq_len, pred_len, shift_pred):
    f = open(filename, 'r').read()
    data = f.split('\n')[:-1] # get rid of the last '' so float(n) works
    data.reverse()
    d = [float(n) for n in data]
    lower = np.min(d)
    upper = np.max(d)
    scale = upper-lower
    normalized_d = [(x-lower)/scale for x in d]

    result = []
    if shift_pred:
        pred_len = 1
    for i in range((len(normalized_d) - seq_len - pred_len)/pred_len):
        result.append(normalized_d[i*pred_len: i*pred_len + seq_len + pred_len])
    
    result = np.array(result)
    print(result.shape)

    row = int(round(0.9 * result.shape[0]))
    train = result[:row, :]
    test = result[row:, :]

    # normally  you should most likely randomly shuffle your data 
    # before splitting it into a training and test set.
    np.random.shuffle(train) 

    x_train = train[:, :-pred_len] # all rows with all columns except the last one
    x_test = test[:, :-pred_len] # rest 10% used for testing

    if shift_pred:
        y_train = train[:, 1:] 
        y_test = test[:, 1:]
    else:
        y_train = train[:, -pred_len:] # all rows with the last column
        y_test = test[:, -pred_len:]
    

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test, lower, scale]

X_train, y_train, X_test, y_test, lower, scale = load_data(symbol + '.txt', seq_len, pred_len, shift_pred)

model = Sequential()

model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(None, 1)), input_shape=(seq_len, 1)))
# return_sequences: whether to return the last output in the output sequence, or the full sequence.
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=True)) # False | True if another LSTM layer is added below
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))

if shift_pred:
    model.add(Dense(units=seq_len))
else:
    model.add(Dense(units=pred_len))

model.add(Activation('linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.fit(
    X_train,
    y_train,
    #callbacks=[keras.callbacks.TensorBoard(log_dir='./logs')],
    batch_size=512,
    epochs=epochs,
    validation_split=0.05)

saver = tf.train.Saver()
saver.save(K.get_session(), 'ckpt/keras_' + symbol + '.ckpt')

predictions = []

correct = 0
total = pred_len*len(X_test)
for i in range(len(X_test)):
    input = X_test[i]
    y_pred = model.predict(input.reshape(1, seq_len, 1))
    #print(y_pred.shape) # (1,20)
    predictions.append(scale * y_pred[0][-1] + lower)
    if shift_pred:
        if y_test[i][-1] >= input[-1][0] and y_pred[0][-1] >= input[-1][0]:
            correct += 1
        elif y_test[i][-1] < input[-1][0] and y_pred[0][-1] < input[-1][0]:
            correct += 1
    else:
        for j in range(len(y_test[i])):
            #print(">>> %f, %f, %f" %(input[-1][0], y_test[i][j], y_pred[0][j]))
            if y_test[i][j] >= input[-1][0] and y_pred[0][j] >= input[-1][0]:
                correct += 1
            elif y_test[i][j] < input[-1][0] and y_pred[0][j] < input[-1][0]:
                correct += 1

print("correct=%d, total=%d, ratio:%.2f%%" %(correct, total, 100*float(correct)/total))

y_test = scale * y_test + lower 
y_test = y_test[:, -1]
xs = [i for i, _ in enumerate(y_test)]
plt.plot(xs, y_test, 'g-', label='true')
plt.plot(xs, predictions, 'r-', label='prediction')
plt.legend(loc=0)
if shift_pred:
    plt.title("%s - epochs=%d, shift_pred=True, seq_len=%d: %d/%d=%.2f%%" %(symbol, epochs, seq_len, correct, total, 100*float(correct)/total))
else:
    plt.title("%s - epochs=%d, lens=%d,%d: %d/%d=%.2f%%" %(symbol, epochs, seq_len, pred_len, correct, total, 100*float(correct)/total))
plt.show()


