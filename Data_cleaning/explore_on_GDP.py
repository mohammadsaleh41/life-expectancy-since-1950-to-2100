# %%
import numpy as np
import pandas as pd
import os
import plotly.express as px
# %%
dirname = os.path.dirname(__file__)
datas_path = dirname[:-13] + "datas/"
datas_path 
# %%
gdp_df = pd.read_csv(datas_path+"Life Expectancy vs GDP 1950-2018.csv")
gdp_df
# %%
gdp_df.isnull().sum()
# %%
gdp_df.iloc[69]["GDP per capita"]
# %%
np.nan
# %%
gdp_df.isnull().iloc[69]["GDP per capita"]
# %%
s = 0
d = 0
for i in range(len(gdp_df)):
    if gdp_df.isnull().iloc[i]["GDP per capita"]:
        if gdp_df.iloc[i]["Year"] == 2019:
            s +=1
        else:
            d+=1

print(s)
print(d)
# %%
len(gdp_df)
# %%
df = pd.read_csv(datas_path + "data.csv")
df
# %%
d = df.groupby("Region, subregion, country or area *").count()["Life Expectancy at Birth, both sexes (years)"]
d
# %%
fig = px.histogram(d , x = d.index , y ="Life Expectancy at Birth, both sexes (years)")
fig.write_html('first_figure.html', auto_open=True)
# %%

