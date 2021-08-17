# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:25:52 2021

@author: ranam
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:19:29 2021

@author: ranam
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from pandas import read_csv
from numpy import concatenate
from matplotlib import pyplot

# Constants
rho_w = 1000 # kg/m3


# Importing data
df = read_csv('RNNDataNew.csv')
#df.drop(df.columns[[0]], axis=1, inplace=True)

# Copying data into arrays
Temperature = df.iloc[:,1].values
LSR = df.iloc[:,2].values
CA = df.iloc[:,3].values
Size = df.iloc[:,4].values
Time = df.iloc[:,0].values
Ramp = df.iloc[:, 7].values
F_X = df.iloc[:,8].values
Ro = df.iloc[:,9].values
logRo = df.iloc[:,10].values
P = df.iloc[:,11].values
logP = df.iloc[:,12].values
H = df.iloc[:,13].values
logH = df.iloc[:,14].values
Acetyl = df.iloc[:,15].values
Woods = df.iloc[:,16:39].values
Yield = df.iloc[:,40].values

# Conditions for temperatures higher than 450 K
for i in range(len(Temperature)):
    if Temperature[i]>450:
        Temperature[i]=450
    if Temperature[i]==0:
        print('There are 0 K temperatures')
        
# Converting acid concentration (proton)
for i in range(len(CA)):
    CA[i] = CA[i]*98.079/2/10

Ho = np.zeros_like(F_X) 
Xo = np.zeros_like(F_X)
Xf = np.zeros_like(F_X)
Hf = np.zeros_like(F_X)

for i in range(len(F_X)):
    Ho[i] = F_X[i]*rho_w/(100*LSR[i])
    Xf[i] = (rho_w*F_X[i]/100)*Yield[i]/(100*LSR[i])
    Hf[i] = Ho[i] - Xf[i]
    
max_t = np.max(Time)
print(max_t)
# Reshaping all X arrays     
Ho = Ho.reshape((Ho.shape[0], 1))
Hf = Hf.reshape((Hf.shape[0], 1))
LSR = LSR.reshape((LSR.shape[0], 1))
CA = CA.reshape((CA.shape[0], 1))
Size = Size.reshape((Size.shape[0],1))
Temp = Temperature.reshape((Temperature.shape[0], 1))
Time = Time.reshape((Time.shape[0], 1))
Ro = Ro.reshape((Ro.shape[0],1))
logRo = logRo.reshape((logRo.shape[0],1))
P = P.reshape((P.shape[0],1))
logP = logP.reshape((logP.shape[0],1))
H = H.reshape((H.shape[0],1))
logH = logH.reshape((logH.shape[0],1))
Acetyl = Acetyl.reshape((Acetyl.shape[0],1))
X = concatenate((Temp, LSR, Size, Ho, CA, Time, Ro, logRo, P, logP, H, logH, Acetyl, Woods), axis=-1)
X = np.nan_to_num(X)


num_features = 36
row1x = (X[1,:])

train_X, test_X, train_Y, test_Y = train_test_split(X, Xf, test_size=0.2, random_state=42)    
#train_X1, test_X1, train_Y, test_Y = train_test_split(X, Hf, test_size=0.2, random_state=1)    

# train_X = train_X1
# test_X = test_X1


max_t_train = np.max(train_X[:,5])
print(max_t_train)

max_t_test = int(np.max(test_X[:,5]))
print(max_t_test)

train_Xnew = []
test_Xnew = []



def new_time(t):
    tenmin=0
    if tenmin==0:
        tenmin=1
    else:
        tenmin = int(t/10)
    return tenmin
    

for i in range(len(train_X)):
    zeros_array = np.zeros(num_features)
    if train_X[i,5]<max_t:
        t = int(train_X[i,5])
        newt = new_time(t)
        rowi_data = np.zeros(num_features)
        for j in range(num_features):
            rowi_data[j] = train_X[i,j]
        # rowi_data[0] = train_X[i,0]
        # rowi_data[1] = train_X[i,1]
        # rowi_data[2] = train_X[i,2]
        # rowi_data[3] = train_X[i,3]
        # rowi_data[4] = train_X[i,4]
        # rowi_data[5] = train_X[i,5]
        # rowi_data[6] = train_X[i,6]
        # rowi_data[7] = train_X[i,7]
        # rowi_data[8] = train_X[i,8]
        # rowi_data[9] = train_X[i,9]
        # rowi_data[10] = train_X[i,10]
        # rowi_data[11] = train_X[i,11]
        # rowi_data[12] = train_X[i,12]
        repeatedt_times = np.tile(rowi_data, (newt,1))
        repeatedt_times = np.transpose(repeatedt_times)
        numzerorepeat = int(max_t/10) - newt 
        zeros = np.tile(zeros_array, (numzerorepeat,1))
        zeros = np.transpose(zeros)
        rowi = concatenate((repeatedt_times, zeros), axis=-1)
        rowi = np.transpose(rowi)
        #rowi = scaler.fit_transform(rowi) 
        train_Xnew.append(rowi)
    else:
        t = int(train_X[i,5])
        newt = new_time(t)
        rowi_data = np.zeros(num_features)
        for j in range(num_features):
            rowi_data[j] = train_X[i,j]
        # rowi_data[0] = train_X[i,0]
        # rowi_data[1] = train_X[i,1]
        # rowi_data[2] = train_X[i,2]
        # rowi_data[3] = train_X[i,3]
        # rowi_data[4] = train_X[i,4]
        # rowi_data[5] = train_X[i,5]
        # rowi_data[6] = train_X[i,6]
        # rowi_data[7] = train_X[i,7]
        # rowi_data[8] = train_X[i,8]
        # rowi_data[9] = train_X[i,9]
        # rowi_data[10] = train_X[i,10]
        # rowi_data[11] = train_X[i,11]
        # rowi_data[12] = train_X[i,12]
        repeatedt_times = np.tile(rowi_data, (int(max_t/10),1))
        #repeatedt_times = scaler.fit_transform(repeatedt_times) 
        #repeatedt_times = np.transpose(repeatedt_times)
        train_Xnew.append(repeatedt_times)


train_Xnew = np.transpose(train_Xnew)
train_Xnewnew = np.dstack(train_Xnew)
X_train =np.dstack( np.transpose(np.transpose(np.transpose(np.transpose(np.transpose(train_Xnewnew))))))
Y_train = train_Y.reshape(train_Y.shape[0], 1)
#print(X_train)

for i in range(len(test_X)):
    zeros_array = np.zeros(num_features)
    if test_X[i,5]<max_t:
        t = int(test_X[i,5])
        newt = new_time(t)
        rowi_data = np.zeros(num_features)
        for j in range(num_features):
            rowi_data[j] = test_X[i,j]
        # rowi_data[0] = test_X[i,0]
        # rowi_data[1] = test_X[i,1]
        # rowi_data[2] = test_X[i,2]
        # rowi_data[3] = test_X[i,3]
        # rowi_data[4] = test_X[i,4]
        # rowi_data[5] = test_X[i,5]
        # rowi_data[6] = test_X[i,6]
        # rowi_data[7] = test_X[i,7]
        # rowi_data[8] = test_X[i,8]
        # rowi_data[9] = test_X[i,9]
        # rowi_data[10] = test_X[i,10]
        # rowi_data[11] = test_X[i,11]
        # rowi_data[12] = test_X[i,12]
        repeatedt_times = np.tile(rowi_data, (newt,1))
        repeatedt_times = np.transpose(repeatedt_times)
        numzerorepeat = int(max_t/10) - newt 
        zeros = np.tile(zeros_array, (numzerorepeat,1))
        zeros = np.transpose(zeros)
        rowi = concatenate((repeatedt_times, zeros), axis=-1)
        rowi = np.transpose(rowi)
        #rowi = scaler.fit_transform(rowi) 
        test_Xnew.append(rowi)
    else:
        t = int(test_X[i,5])
        rowi_data = np.zeros(num_features)
        for j in range(num_features):
            rowi_data[j] = test_X[i,j]
        # rowi_data[0] = test_X[i,0]
        # rowi_data[1] = test_X[i,1]
        # rowi_data[2] = test_X[i,2]
        # rowi_data[3] = test_X[i,3]
        # rowi_data[4] = test_X[i,4]
        # rowi_data[5] = test_X[i,5]
        # rowi_data[6] = test_X[i,6]
        # rowi_data[7] = test_X[i,7]
        # rowi_data[8] = test_X[i,8]
        # rowi_data[9] = test_X[i,9]
        # rowi_data[10] = test_X[i,10]
        # rowi_data[11] = test_X[i,11]
        # rowi_data[12] = test_X[i,12]
        repeatedt_times = np.tile(rowi_data, (int(max_t/10),1))
        #repeatedt_times = scaler.fit_transform(repeatedt_times)
        #repeatedt_times = np.transpose(repeatedt_times)
        test_Xnew.append(repeatedt_times)



train_Xnew = np.transpose(train_Xnew)
train_Xnewnew = np.dstack(train_Xnew)
#X_train = np.dstack( np.transpose(np.transpose(np.transpose(np.transpose(np.transpose(train_Xnewnew))))))
#train_Xnew= scaler.fit_transform(train_Xnew) 
X_train = train_Xnew
Y_train = train_Y.reshape(train_Y.shape[0], 1)
#Y_train = scaler.fit_transform(Y_train)
#print(X_train)

test_Xnew = np.transpose(test_Xnew)
test_Xnewnew = np.dstack(test_Xnew)
X_test =np.dstack( np.transpose(np.transpose(np.transpose(np.transpose(np.transpose(test_Xnewnew))))))
#X_test = scaler.fit_transform(X_test)
Y_test = test_Y.reshape(test_Y.shape[0], 1)
#Y_test = scaler.fit_transform(Y_test)
dropout= 0.2


scalers = {}
for i in range(X_train.shape[1]):
    scalers[i] = RobustScaler()
    X_train[:, i, :] = scalers[i].fit_transform(X_train[:,i,:])
    
for i in range(X_test.shape[1]):
    X_test[:,i,:] = scalers[i].transform(X_test[:,i,:])

#regularizer = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, 0.01), L1L2(l1=0.01, l2=0.01)]
regularizer = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.0001, l2=0.0), L1L2(l1=0.0, l2=0.0001), L1L2(l1=0.0001, l2=0.0001)]
for reg in regularizer:
  model = Sequential()
  model.add(Masking(mask_value=0, input_shape=(int(max_t/10), num_features)))
  model.add(LSTM(50, use_bias=True, return_sequences=False, activity_regularizer=reg, bias_regularizer=reg, kernel_regularizer=reg))
# model.add(LSTM(30, return_sequences=False))
  model.add(Dropout(0.01))
  model.add(Dense(units=64, activation='sigmoid'))
# model.add(Dropout(0.001))
  model.add(Dense(units=48, activation='sigmoid'))        
  model.add(Dense(units=48, activation='sigmoid'))
  model.add(Dense(1, activation='linear'))
  opt = Adam()
  model.compile(loss='mean_squared_error', optimizer=opt)
  print(model.summary())
  
  

  history = model.fit(X_train, Y_train, epochs=1000, batch_size=12, validation_data=(X_test, Y_test), verbose=0)
  
  #pyplot.plot(history.history['loss'], label = 'train')
  #pyplot.plot(history.history['val_loss'], label='test')
  #pyplot.legend()
  #pyplot.show()

  yhat = model.predict(X_train)
  MAE_train = metrics.mean_absolute_error(Y_train, yhat)
  print('MAE train for xylose conc.: ', MAE_train)
  Yieldtrainpred = np.zeros_like(yhat)
  Yieldtrain = np.zeros_like(yhat)
  Fxtrain = np.zeros_like(yhat)
  Xftrain = np.zeros_like(yhat)
  for i in range(len(yhat)):
      Fxtrain[i] = train_X[i,3]*(100*train_X[i,1])/rho_w
    #Xftrain[i] = train_X[i,3] - yhat[i]
      Yieldtrainpred[i] = yhat[i]*(100*train_X[i, 1])/(rho_w*Fxtrain[i]/100)
      Yieldtrain[i] = train_Y[i]*(100*train_X[i, 1])/(rho_w*Fxtrain[i]/100)

  MAE_trainyield = metrics.mean_absolute_error(Yieldtrain, Yieldtrainpred)
  print('MAE train for yield: ', MAE_trainyield)

  ypredtest = model.predict(X_test)
  MAE_test = metrics.mean_absolute_error(Y_test, ypredtest)
  print('MAE test for xylose conc.: ', MAE_test)
  Yieldtestpred = np.zeros_like(ypredtest)
  Yieldtest = np.zeros_like(ypredtest)
  Fxtest = np.zeros_like(ypredtest)
  Xftest = np.zeros_like(ypredtest)
  for i in range(len(ypredtest)):
      Fxtest[i] = test_X[i,3]*(100*test_X[i,1])/rho_w
    #Xftest[i] = test_X[i,3] - ypredtest[i]
      Yieldtestpred[i] = ypredtest[i]*(100*test_X[i, 1])/(rho_w*Fxtest[i]/100)
      Yieldtest[i] = test_Y[i]*(100*test_X[i, 1])/(rho_w*Fxtest[i]/100)

  MAE_testyield = metrics.mean_absolute_error(Yieldtest, Yieldtestpred)
  print('MAE test for yield: ', MAE_testyield)
