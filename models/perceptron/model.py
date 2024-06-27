# %%
import numpy as np
import pandas as pd

# %%
x  = np.array([[2 ,1] , [2 , 4],[ 3 , 5]])
w = np.array([4 , 5 , 3])
w.T.dot(x)

# %%
class Perceptron(object):
    def __init__(self , input_size , lr = 1 , epochs = 100):
        self.W = np.zeros(input_size +1)
        # یک دونه اضافه تر به خاظر بایاس هستش
        self.epochs = epochs
        self.lr = lr
    def activation_fn(self , x):
        if x>=0:
            return 1
        else:
            return 0
    def predict(self , x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    def fit (self , X , d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i] , 0 , 1)
                y = self.predict(x)
                print("x : ",x)
                
                print("y : " , y)
                
                e = d[i] - y
                self.W = self.W + self.lr * e * x
                print("e : ", e)
                print("W : ", self.W)
# %%
X = np.array(
   [[0 , 0],
    [0 , 1],
    [1 , 0],
    [1 , 1]]
)
Y = np.array([0 , 0 , 0 , 1])

model = Perceptron(2)
model.fit (X , Y)
print(model.W)
# %%
