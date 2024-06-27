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
dirname = os.path.dirname(__file__)
Datasets_path = dirname[:-15] + "Datasets/"
# %%
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Do you want to add filters?")
    if not modify:
        return df
    df = df.copy()

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
    return df



# %%
df = pd.read_csv(Datasets_path+"ModelData.csv")
if "Unnamed: 0" in df.columns:
    df.drop(columns= "Unnamed: 0", inplace=True)

df_2000 = df[df["Year"]==2000][["Region, subregion, country or area *","Life Expectancy at Birth, both sexes (years)"]]
df_2021 = df[df["Year"]==2021][["Region, subregion, country or area *","Life Expectancy at Birth, both sexes (years)"]]
new_df = df_2000.merge(df_2021 , on = "Region, subregion, country or area *")
#new_df["Region, subregion, country or area *"] = df_2000[["Region, subregion, country or area *"]]
# %%
new_df["delta"] = new_df["Life Expectancy at Birth, both sexes (years)_y"]-new_df["Life Expectancy at Birth, both sexes (years)_x"]
new_df = new_df[["Region, subregion, country or area *","delta"]].sort_values(["delta"],ascending = False)
new_df[:10]

# %%
