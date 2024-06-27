# %%
import numpy as np
import pandas as pd
import os
import plotly.express as px
# %%
dirname = os.path.dirname(__file__)
Datasets_path = dirname[:-13] + "Datasets/"
datas_path = dirname[:-13] + "datas/"
datas_path 
# %%
df = pd.read_csv(Datasets_path+"CountryInfo.csv")
df_l = pd.read_csv(datas_path+"Life Expectancy vs GDP 1950-2018.csv")
df_l
# %%

# %%
df_wb = pd.read_csv(datas_path+"API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4888903.csv") 
df_wb
# %%

df
# %%
n_wb = pd.melt(df_wb.set_index("Country Code").drop(columns = ["Country Name"]), ignore_index=False, var_name="Year" , value_name="GDP per capita")
n_wb["Year"] = n_wb["Year"].astype(int)
df["Year"] = df["Year"].astype(int)

# %%
new_df = df.merge(n_wb ,how = "left", left_on= ["ISO3 Alpha-code" ,"Year"] ,  right_on=["Country Code","Year"] , indicator=True)
new_df
# %%
# %%
new2_df =  new_df.merge(df_l ,how  = "left" , left_on = ["Region, subregion, country or area *" ,"Year"] , right_on = ["Country","Year"])
new2_df
# %%
new2_df.to_csv("gdp_added.csv")
# %%
