# %%
# وارد کردن کتابخوانه‌ها
import os
import plotly as px
import numpy as np
import pandas as pd
import missingno as msno
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

# %%
# گرفتن آدرس‌ها
dirname = os.path.dirname(__file__)
Datasets_path = dirname[:-21] + "Datasets/"
# برای اجرای کد در ویندوز خط بالای اینجا باید به خط پایینی تغییر کند.
# Datasets_path = dirname[:-21] + "Datasets\\"
# %%
# فراخوانی جدول نهایی آماده شده
df = pd.read_csv(Datasets_path+"DataWithFullGDP.csv")
# اگر جدول با ایندکس ذخیره شده باشه ایندکس حذف می‌شه
if "Unnamed: 0" in df.columns:
    df.drop(columns= "Unnamed: 0", inplace=True)
df
# %%
# چک کردن وجود null
df.isnull().sum()
# دروغ می‌گه :)
# %%
df.info()
# %%
# در صورتی که داده‌های اولیه‌ای که خیلی تمیز نبودن رو به این قسمت وارد کنیم می‌تونه خوب مدیریتش کنه.
# در اینجا ما داده‌‌ها رو سعی می‌کنیم float کنیم و در صورت گرفتن خطا ستون های مورد دار رو پیدا می‌کنیم.

#df = pd.read_csv(Datasets_path+"CountryInfo.csv")
list_col_err =[]
for col in df.columns[2:]:
    try:
        df[col] = df[col].astype(float)
    except ValueError as err:
        if "could not convert string to float: '...'" in str(err):
            list_col_err.append(col)
        else:
            print(err)

len(df.columns[2:]),len(list_col_err)
#اینجا می‌تونیم بفهمیم چه تعداد از کل ستون‌ها غیر قابل تبدیل هستن.
# %%
# اینجا داده‌هایی که به شکل سه نقطه بدون رو به nan تبدیل می‌کنیم.
# می‌شه این‌جا رو تغییر داد به صورتی که هر داده‌ای که به عدد تبدیل نشد رو به nan تبدیل کنه.
# به شرطی که قبلش دیگه داده حرفی نداشته باشیم و سعی کرده باشیم همه رو عددی کنیم.
for col in list_col_err:
    for j in range(len(df)):
        if df.at[j , col] == '...':
            df.at[j , col] = np.nan


# %%
# حالا می‌فهمیم اوضاع واقعا چه جوریه.
df.isnull().sum()

# %%
# مصور سازی داده‌های از دست رفته
msno.matrix(df)
# %%
# این ستون داده از دست رفته زیادی داره.
df.columns[14]
# %%
df[df[df.columns[5]].isnull()]
# اگر توی غیر از اون ستون 'Population Annual Doubling Time (years)' مشکلی می‌بود به کمک این قسمت می‌تونستیم اون بخش‌ها رو پیدا کنیم.


# %%
# فعلا هر ردیفی که داده از دست رفته داره رو حذف می‌کنیم.
df.dropna(inplace=True)
# %%
# ستون اسم و کد کشور رو از روی مجموعه داده‌ها حذف می‌کنیم.
df.drop(columns=["Region, subregion, country or area *","ISO3 Alpha-code"], inplace=True)
# %%
# اینجا داده‌های ترین و تست رو باید درست کنیم.
# کاری که این‌جا هنوز انجام نشده اینه که باید از روی دیتا فریم مون مجموعه X و Y رو مشخص کنیم.
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = df.drop(columns=[اون ستون مهم که باید حدس زده بشه])
Y = df[[اون ستون مهم که باید حدس زده بشه]]
train_X , test_X, train_Y , test_Y = train_test_split(X ,Y , test_size=.2, random_state=313)
train_X , valid_X , train_Y , valid_Y = train_test_split(train_X,train_Y, test_size=.2, random_state=313)

# %%
train

# %%
# 2. define model
init = K.initializers.RandomUniform(seed=1)
simple_sgd = K.optimizers.SGD(learning_rate=0.010)#پارامتر نرخ یادگیری اسم جدیدتری دارد
# توی لینک زیر یک مقدار متفاوت هستش
# https://keras.io/api/optimizers/sgd/
# بنظر می‌رسه درستش اینه:
# simple_sgd = K.optimizers.experimental.SGD(learning_rate=0.1)

# %%
# این لایه ورودیش ۱۳ تا ورودی داره باید ببینیم در نهایت چند تا ورودی به مدل مون باید داشته باشیم اون وقت باید اون input_dim رو ویرایش کرد.
model = K.models.Sequential()
model.add(K.layers.Dense(units=10, input_dim=57, kernel_initializer=init, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=10, activation='tanh')) # hidden layer
model.add(K.layers.Dense(units=1, activation=None))
model.compile(loss='mean_squared_error', optimizer = simple_sgd , metrics=['mse'])

# %%
# 3. train model
batch_size= 8
max_epochs = 500

print("Starting training ")
h = model.fit(train_X, train_Y, batch_size=batch_size, epochs=max_epochs, verbose=1)
print("Training finished \n")


# %%
