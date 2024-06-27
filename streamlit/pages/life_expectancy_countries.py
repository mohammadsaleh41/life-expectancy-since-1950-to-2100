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
countries = pd.read_csv(Datasets_path+'average-latitude-longitude-countries.csv')
if "Unnamed: 0" in df.columns:
    df.drop(columns= "Unnamed: 0", inplace=True)
df.dropna(inplace = True)



# %%
#df = df.merge(countries , right_on ="Country" , left_on = "Region, subregion, country or area *" , how = "left")

#df_g = df.groupby("Region, subregion, country or area *").sum()

c = st.slider('ستون چندم؟', 3, len(df.columns)-1, 46)
st.write("ستون "+df.columns[c])

fig = px.scatter_geo(df, locations="ISO3 Alpha-code", color="Region, subregion, country or area *", size=df.columns[c],
                     animation_frame="Year")
st.plotly_chart(fig, use_container_width=True)
# ISO3 Alpha-code
