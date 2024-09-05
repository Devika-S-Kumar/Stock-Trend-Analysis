import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start='2010-01-01'
end='2099-12-31'

st.title('STOCK TREND PREDICTION')

user_input = st.text_input('Enter Stock Ticker','AAPL')

df = yf.download(user_input,start,end)

st.subheader('Data from 2010-2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


tr=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
tst=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(tst.shape)
print(tr.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
tr_a=scaler.fit_transform(tr)

xtrain=[]
ytrain=[]

for i in range(100,tr_a.shape[0]):
    xtrain.append(tr_a[i-100:i])
    ytrain.append(tr_a[i,0])

xtrain,ytrain = np.array(xtrain), np.array(ytrain)

#loading model
model=load_model('keras_model.h5')


#testing part

past=tr.tail(100)
finaldf=pd.concat([past,tst],ignore_index=True)
inpt=scaler.fit_transform(finaldf)
xtest=[]
ytest=[]

for i in range(100,inpt.shape[0]):
    xtest.append(inpt[i-100:i])
    ytest.append(inpt[i,0])

xtest,ytest = np.array(xtest), np.array(ytest)

y_predict = model.predict(xtest)
scalar=scaler.scale_
sf=1/scalar[0]
y_predict = y_predict*sf
ytest = ytest*sf

st.subheader('Orginal Price vs Predicted Price')
fig3=plt.figure(figsize = (12,6))
plt.plot(ytest,'b',label='Orginal Price')
plt.plot(y_predict,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)