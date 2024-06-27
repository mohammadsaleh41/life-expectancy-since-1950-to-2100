import os
import numpy as np
import pandas as pd
import plotly as px
import tensorflow as tf
import missingno as msno
from tensorflow import keras as K
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


dirname = os.path.dirname(__file__)
# Windows
datasets_path = dirname[:-8] + 'Datasets\\'
# Linux
# datasets_path = dirname[:-8] + 'Datasets/'
df = pd.read_csv(datasets_path + 'ModelData.csv').drop(columns=['Unnamed: 0',
                                                                'Region, subregion, country or area *',
                                                                'ISO3 Alpha-code',
                                                                'Parent code',
                                                                'Location code'])


df.info()


msno.matrix(df)


df.dropna(inplace=True)
msno.matrix(df)


X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
train_X , test_X, train_Y , test_Y = train_test_split(X, Y,
                                                      test_size=0.2,
                                                      random_state=313)
train_X , valid_X , train_Y , valid_Y = train_test_split(train_X, train_Y,
                                                         test_size=0.2,
                                                         random_state=313)


init = K.initializers.RandomUniform(seed=1)
simple_sgd = K.optimizers.SGD(learning_rate=0.010)


model = K.models.Sequential()
model.add(K.layers.Dense(units=10, input_dim=44, kernel_initializer=init, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=10, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=1, activation=None))
model.compile(loss='mean_squared_error', optimizer = simple_sgd , metrics=['mse'])


batch_size= 8
max_epochs = 500
print("Starting training ")
h = model.fit(train_X, train_Y, batch_size=batch_size, epochs=max_epochs, verbose=1)
print("Training finished \n")


def my_accuracy(model, data_x, data_y, pct_close):
  num_correct = 0; num_wrong = 0
  n = len(data_x)
  for i in range(n):
    predicted = model.predict(np.array([data_x[i]], dtype=np.float32))
    actual = data_y[i]
    if np.abs(predicted[0][0] - actual) < np.abs(pct_close * actual):
      num_correct += 1
    else:
      num_wrong += 1
  return (num_correct * 1.0) / (num_correct + num_wrong)


acc = my_accuracy(model, train_X, train_Y, 0.15)
print("Overall accuracy (wthin 15%%) on training data = %0.4f" % acc)


acc = my_accuracy(model, test_X, test_Y, 0.15)
print("Overall accuracy on test data = %0.4f \n" % acc)


eval = model.evaluate(train_X, train_Y, verbose=1)
print("Overall loss (mse) on training data = %0.6f" % eval[0])


eval = model.evaluate(test_X, test_Y, verbose=1)
print("Overall loss (mse) on test data = %0.6f" % eval[0])
