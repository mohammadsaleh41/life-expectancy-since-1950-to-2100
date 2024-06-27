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
# %%
df_wb = pd.read_csv(datas_path+"API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4888903.csv") 
df_wb
# %%
df

# %%
khali = 0
gdp_label = "GDP per capita"
df[gdp_label] = 0

# %%
#df.loc[[5],[gdp_label]] = 5

# %%

# %%
df.loc[5][gdp_label] ==0
# %%
for i in range(len(df)):
    c = df.iloc[i]["Region, subregion, country or area *"]
    year = df.iloc[i]["Year"]
    if year<1960:
        for j in range(len(df_l)):
            if  df_l.iloc[j]["Year"] == year:
                if df_l.iloc[j]["Country"] == c:
                    df.loc[[i],[gdp_label]] = df_l.iloc[j]["GDP per capita"]
                    break
# %%    
    else:
        for j in range(len(df_wb)):
            if c == df_wb.iloc[j]["Country Name"]:
                if not( df_wb.isnull().iloc[j][str(int(year))]):
                    df.loc[[i],[gdp_label]] = df_wb.iloc[j][str(int(year))]
                else:
                    break
        if 0 != df.loc[i][gdp_label]:
            continue
        for j in range(len(df_l)):
            if  df_l.iloc[j]["Year"] == year:
                if df_l.iloc[j]["Country"] == c:
                    df.loc[[i],[gdp_label]] = df_l.iloc[j]["GDP per capita"]
                    break
# %%

df.to_csv(dirname+"/gdp_added.csv")
# %%


# %%
