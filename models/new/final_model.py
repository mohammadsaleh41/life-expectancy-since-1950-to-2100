# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
from timeit import default_timer as timer
# %%
class nor():
    def __init__(self , arr):
        self.arr = arr.T
        self.makhrajha = []
        self.vasatha = []
        for i in range(len(self.arr)):
            n_arr , vasat ,makhraj = self.first_normalize(arr[i])
            self.makhrajha.append(makhraj)
            self.vasatha.append(vasat)


        
    def first_normalize(self, arr):
        max = np.max(self.arr)
        min = np.min(self.arr)
        vasat = (min+max)/2
        n_arr = 2*((arr)-vasat)/(max-min)
        return n_arr , vasat , max-min
    def normalize(self, ar):
        t_arr = ar.T
        for i in range(len(t_arr)):
            t_arr[i] = 2*((t_arr[i])-self.vasatha[i])/(self.makhrajha[i])
        return t_arr.T
    def unormalize(self , ar):
        t_arr = ar.T
        for i in range(len(t_arr)):
            t_arr[i] = ((t_arr[i]*self.makhrajha[i])/2)+self.vasatha[i]
        return t_arr.T
# %%
dirname = os.path.dirname(__file__)
Datasets_path = dirname[:-10] + "Datasets/"
Datasets_path

# %%
model_data = pd.read_csv(Datasets_path+'ModelData.csv').drop(columns=['Unnamed: 0'])
estimate_data = pd.read_csv(Datasets_path+'ModelEstimationData.csv')
countries = list(pd.read_csv(Datasets_path+'CountryNames.csv').iloc[:, 0])

# %%
model_data.dropna(inplace=True)

# %%
X = model_data.iloc[:, 2:-1].values
Y = model_data.iloc[:, -1].values


normalizer_X = nor(X)
normalizer_Y = nor(Y)

X = normalizer_X.normalize(X)
Y = normalizer_Y.normalize(Y)

# %%

train_X , test_X, train_Y , test_Y = train_test_split(X, Y,
                                                      test_size=0.2,
                                                      random_state=313)
train_X , valid_X , train_Y , valid_Y = train_test_split(train_X, train_Y,
                                                         test_size=0.2,
                                                         random_state=313)



# %%

initializer = tf.keras.initializers.TruncatedNormal(mean=1., stddev=3.)
# %%
print("-*-*-*-\n-*-*-*-\n-*-*-*-\n")
config = tf.compat.v1.ConfigProto(log_device_placement=True )

sess = tf.compat.v1.Session(config=config) 

print("-*-*-*-\n-*-*-*-\n-*-*-*-\n")
model = K.models.Sequential()

model.add(K.layers.Dense(units=30, input_dim=44, kernel_initializer=initializer, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=30, kernel_initializer=initializer, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=30, kernel_initializer=initializer, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=30, kernel_initializer=initializer, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=30, kernel_initializer=initializer, activation='tanh')) # hidden layer

model.add(K.layers.Dense(units=1, kernel_initializer=initializer))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# %%



# %%
max_epochs = 3000

print("Starting training ")
start = timer()
h = model.fit(train_X, train_Y,batch_size=4000 ,epochs=max_epochs, verbose=0)
print("train time: ", timer()-start)
print("Training finished \n")

# %%

model2 = LinearRegression()
# %%
model2.fit(train_X, train_Y)

# %%
pred_df = pd.read_csv(Datasets_path+'ModelEstimationData.csv').drop(columns=['Region, subregion, country or area *',
                                                                        'ISO3 Alpha-code'])
pred_df = pred_df[pred_df['Year'] == 2022]
pred_df.dropna(inplace=True)
# %%
unknown = pred_df.iloc[:, :].values

# %%
#print(model2.predict(np.array(train_X[-5:-1])))

# %%
r_sq = model2.score(test_X, test_Y)
r_sq

# %%
r_sq = model2.score(valid_X, valid_Y)
r_sq
# %%
r_sq = model2.score(train_X, train_Y)
r_sq

# %%

unknown = normalizer_X.normalize(unknown)

predicted = model2.predict(unknown)
#print(round(predicted[0]))
predicted = normalizer_Y.unormalize(predicted)
#for i in range(len(predicted)):
#    print(round(predicted[i]))
# %%
round(unknown[2][3])


# %%
from sklearn.metrics import r2_score
predicted = model2.predict(test_X)

r2_score(predicted , test_Y)


# %%
predicted = model.predict(test_X)
r2_score(predicted , test_Y)

# %%
predicted = model.predict(unknown)

predicted = normalizer_Y.unormalize(predicted)
#for i in range(len(predicted)):
#    print(round(predicted[i][0],3))

# %%
# %%
t = time.localtime()
predicted = model.predict(test_X)
model.save("trained_model_"+str(r2_score(predicted , test_Y))+str(t.tm_year)+"_"+str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"_"+str(t.tm_min)+".h5")
# %%

