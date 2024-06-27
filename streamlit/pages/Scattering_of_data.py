# %%
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import streamlit as st
import os
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import math

dirname = os.path.dirname(__file__)
Datasets_path = dirname[:-15] + "Datasets/"
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
df = pd.read_csv(Datasets_path+'ModelData.csv').drop(columns=['Unnamed: 0'])
if "Unnamed: 0" in df.columns:
    df.drop(columns= "Unnamed: 0", inplace=True)
df.dropna(inplace = True)


X = df.iloc[:, 2:-1].values
Y = df.iloc[:, -1].values


normalizer_X = nor(X)
normalizer_Y = nor(Y)
X = df.iloc[:, 2:-1].values
X_n = normalizer_X.normalize(X)
Y = normalizer_Y.normalize(Y)



#print([math.tanh(normalizer.first_normalize(n)[0][i]) for i in range(len(n))])
X_n = X_n.T
X = X.T

# %%
c = st.slider('ستون چندم؟', 0, 43, 0)
X = df.iloc[:, 2:-1].values
X = X.T
st.write("ستون "+df.columns[c])
fig = px.scatter(X , x = X[c], y = [math.tanh(x) for x in X_n[c]], template='plotly_dark' , labels={"x":"تبدیل شده بعد از تابع فعال ساز" , "y": "قبل از وارد شدن به مدل"})
st.plotly_chart(fig, use_container_width=True)
fig=px.histogram(X,x=X[c] ,title=df.columns[c]+" پراکندگی متفیر ")
st.plotly_chart(fig, use_container_width=True)
# %%

# %%
