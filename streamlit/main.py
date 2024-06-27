# %%
# وارد کردن کتابخوانه‌ها

import streamlit as st
import os
import plotly as px
import numpy as np
import pandas as pd
import plotly.figure_factory as ff


# %%
# گرفتن آدرس‌ها
dirname = os.path.dirname(__file__)
Datasets_path = dirname[:-9] + "Datasets/"
# برای اجرای کد در ویندوز خط بالای اینجا باید به خط پایینی تغییر کند.
# Datasets_path = dirname[:-21] + "Datasets\\"
# %%
# فراخوانی جدول نهایی آماده شده
df = pd.read_csv(Datasets_path+"ModelData.csv")
if "Unnamed: 0" in df.columns:
    df.drop(columns= "Unnamed: 0", inplace=True)
df
# %%
