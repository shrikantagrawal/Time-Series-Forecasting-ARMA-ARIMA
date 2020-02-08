# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:13:36 2020

@author: Shrikant Agrawal
"""

# We can predict what may happen in the future by using ARIMA model

import numpy as np
import pandas as pd
import statsmodels.api as sm    # for ARIMA
import matplotlib.pyplot as plt
%matplotlib inline

# Import Dataset
df=pd.read_csv('perrin-freres-monthly-champagne-.csv')

df.tail()

# Row number 105 and 106 has missing values, remove it
df.drop(105,axis=0,inplace=True)
df.drop(106,axis=0,inplace=True)

# Rename column name
df.columns=['Month','Sales per month' ]

df.head()

# Change Date Format so that when we plot it on X axis it will show only years
df['Month']=pd.to_datetime(df['Month'])

df.head()

# For plotting mechanism we don't need index column - Replace index column with month
df.set_index('Month',inplace=True)

df.head()

# Based on date we can predict sales per month
df.plot()

# Run SARIMAX
model=sm.tsa.statespace.SARIMAX(df['Sales per month'],order=(1, 0, 0),seasonal_order=(1,1,1,12))
results=model.fit()

# Forecast the result by using the predictor
df['forecast']=results.predict(start=90,end=103,dynamic=True)  # predict from 90th row to 103 row
df[['Sales per month','forecast']].plot(figsize=(12,8))  # Or simply run df.plot()

# Lets predict it for row 15 to 30
df['forecast']=results.predict(start=15,end=30,dynamic=True)  # predict from 90th row to 103 row
df[['Sales per month','forecast']].plot(figsize=(12,8))

# Crete dataframe for future period for 24 months ie 2 years
from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

# Future dataset do not have any values 
future_datest_df

#Concatenate future data value with our present dataset
future_df=pd.concat([df,future_datest_df])

# It predicts future sale for 2 years
future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales per month', 'forecast']].plot(figsize=(12, 8))
