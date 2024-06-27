# %%
import numpy as np
# %%
class MultiLayerPerceptron(object):
    def __init__(self,layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.activation_fn = sigmoid
        self.d_activatoin_fn = d_sigmoid
        self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(layers[:-1],layers[1:])]


    def sigmoid(z):
        return 1.0/(1+np.exp(-z))
    def d_sigmoid(z):
        return sigmoid(z)*(1-sigmoid(z))

# %%
model = MultiLayerPerceptron([5,2,7])

# %%
