# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:13:06 2021

@author: ranam
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Evaluation of Simplified Kinetic Model by Edward Wang and edited by Rana A. Barghout

from sklearn import metrics
import pandas as pd
import numpy as np
import math
from scipy import optimize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking, Dropout
from tensorflow.keras.optimizers import SGD
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

R = 8.314e-3

## Extracting Data from CSV File 
df = pd.read_csv('PreparedDataWithSourceEdited.csv')
df = df[df.Delete < 1]
df = df.reset_index()
factors = ['Ro', 'logRo', 'P', 'logP', 'H', 'logH']
sequence = ['New Series?']
labels_to_drop_front = [['Temp'], ['LSR'], ['CA'], ['Size'], ['IsoT'], ['F_X'], ['TotalT']]
labels_to_drop_back = [['Ro', 'logRo'],  ['P', 'logP'], ['H', 'logH'], factors  ]
labels_to_drop_all = labels_to_drop_front + labels_to_drop_back
labels_to_drop_front_flat = [item for sublist in labels_to_drop_front for item in sublist]
labels_to_scale = labels_to_drop_front_flat + factors + sequence
X_nonscaled = df[labels_to_scale]
finalCols = labels_to_scale
labels_short = [factors]
X_all = pd.concat([X_nonscaled], ignore_index=True,axis=1)
X_all.columns = finalCols 

# Looking at distribution of Yield Data
Y = (df['Yield'])
y_train = Y.tolist()
YIELD_train = y_train
#X_train, X_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.2, random_state=1)


Temp_train = X_all['Temp'].tolist()
LSR_train = X_all['LSR'].tolist()
CA_train = X_all['CA'].tolist()
F_X_train = X_all['F_X'].tolist()
Size_train = X_all['Size'].tolist()
Ro_train = X_all['Ro'].tolist()
logRo_train = X_all['logRo'].tolist()
P_train = X_all['P'].tolist()
logP_train = X_all['logP'].tolist()
H_train = X_all['H'].tolist()
logH_train = X_all['logH'].tolist()
IsoT_train = X_all['IsoT'].tolist()
TotalT_train = X_all['TotalT'].tolist()
Series_train = X_all['New Series?'].tolist()

rows = [] 

for i in range(len(Series_train)):
    if i==0:
        rows.append(i)
    if Series_train[i]==1:
        rows.append(i)

Temp_set = np.zeros(len(rows)).tolist()
LSR_set = np.zeros(len(rows)).tolist()
CA_set = np.zeros(len(rows)).tolist()
Fx_set = np.zeros(len(rows)).tolist()
IsoT_set = np.zeros(len(rows)).tolist()
Weights_set = np.zeros(len(rows)).tolist()

for i in range(len(Temp_set)):
    index = rows[i]
    Temp_set[i] = Temp_train[index]
    LSR_set[i] = LSR_train[index]
    CA_set[i] = CA_train[index]
    Fx_set[i] = F_X_train[index]

Yield_set = []
TotalT_set = []

# Creating list of lists for yield and time for each experimental set    
for i in range(len(rows)):
    index = rows[i]
    settime = []
    setyield = []
    intg = index
    if i==(len(rows)-1):
        break
    while intg < (rows[i+1]):
        settime.append(TotalT_train[intg])
        setyield.append(y_train[intg])
        if intg==(len(rows)):
            break
        else:
            intg = intg + 1
    TotalT_set.append(settime)
    Yield_set.append(setyield)

data1 = intg
data2 = data1 + 1
data3 = data2 + 1

lastset = range(intg,len(IsoT_train))

settime = []
setyield = []
for i in lastset:
    datapointime = TotalT_train[i]
    datapointyield = y_train[i]
    settime.append(datapointime)
    setyield.append(datapointyield)


TotalT_set.append(settime)
Yield_set.append(settime)

# Creating a weights array that is proportional to the number of datapoints in each experiment
for i in range(len(TotalT_set)):
    array = TotalT_set[i]
    length = len(array)
    Weights_set[i] = length
    
YieldSol_train = np.zeros(len(CA_set))

# Defining all kinetic functions

def getK (A, E, Ca, m, T):
    # A in min^(-1); E in kJ/mol, Ca in %, and T in K
    if T>450:
        T=450
    if Ca>0: 
        return A*Ca**m*math.exp(-E/(8.3143e-3*T))
    else:
        return A*math.exp(-E/(8.3143e-3*T))
    

def getX(k1,k2, H0, t):
    num = -k1*H0*(math.exp(-k1*t)-math.exp(-k2*t))
    denom = k1 - k2
    return num/denom

# Function calculating the yield and error     
def get_error(params):
    Yield_solset = np.zeros_like(y_set)
    for j in range(len(y_set)):
        A1, E1, m1, A2, E2, m2 = params
        t = Time_set[j]
        if CA==0:
            k1=A1*math.exp(-E1/(R*T))
            k2=A2*math.exp(-E2/(R*T))
        else:
            k1=A1*CA**(m1)*math.exp(-E1/(R*T))
            k2=A2*CA**(m2)*math.exp(-E2/(R*T))
        X_sol = (-k1*H0*(math.exp(-k1*t)-math.exp(-k2*t)))/(k1-k2)
        Yield_solset[j] = 100*X_sol*LSR/(1000*(Fx/100))
    error = metrics.mean_absolute_error(y_set, Yield_solset)
    return error

kinetic_params = []
errorsab = [] 

# START OF STEP 1 


# Loop with optimization for each experimental set 
for i in range(len(CA_set)):
    CA = CA_set[i]
    T = Temp_set[i]
    LSR = LSR_set[i]
    Fx = Fx_set[i]
    H0 = (Fx/100)/(LSR+1)*1000
    y_set = np.zeros_like(Yield_set[i])
    y_set = Yield_set[i]
    Time_set = TotalT_set[i]
    print('Index: ', i)
    #print('Yield set: ', y_set)
    initialGuessParams = [4.67e16, 142.58, 1.75, 6.51e16, 155.36,1]
    output = optimize.fmin(get_error, initialGuessParams, maxiter=10000, full_output=1)
    minimum = output[0]
    errors = output[1]
    #print('Calculation Error is: ', errors)
    actual_error = get_error(minimum)
    bestA1, bestE1, bestm1, bestA2, bestE2, bestm2 = minimum
    kinetic_params.append(minimum)
    errorsab.append(actual_error)
    
print("Kinetic Parameters: ", kinetic_params)
print('Error: ', errors)


# START OF STEP 2


# Creating Y array with predicted kinetic parameters for NN
Y = kinetic_params
Y = np.transpose(Y)
Y = np.transpose(Y)
X_new = np.zeros((287,15))


# Creating X array and splitting into training and testing dataset
feats = range(14)

for i in range(len(Temp_set)):
    index=rows[i]
    X_new[i,0]=Temp_train[index]
    X_new[i,1]=LSR_train[index]
    X_new[i,2]=CA_train[index]
    X_new[i,3]=F_X_train[index]
    X_new[i,4]=Size_train[index]
    X_new[i,5]=Ro_train[index]
    X_new[i,6]= logRo_train[index]
    X_new[i,7]=P_train[index]
    X_new[i,8]=logP_train[index]
    X_new[i,9]=H_train[index]
    X_new[i,10]=logH_train[index]
    X_new[i,11]=IsoT_train[index]
    X_new[i,12]=TotalT_train[index]
    X_new[i,13]=Weights_set[i]
    X_new[i,14]=YIELD_train[index]
    
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, random_state=1)

# Extracting data from X and redefining
weights=X_train[:,13]
yield_trainset = X_train[:,14]
yield_testset = X_test[:,14]
X_train = X_train[:,:13]
dimension=X_train.shape[1]
X_test = X_test[:,:13]
print(dimension)

# Defining NN parameters
print('Starting Execution of NN')
cvscores = []
trainingscores =[]
best_lr = 0.005
best_bs = 64
dropout=0.001
epoch=3000

# Start of NN
model = Sequential()
model.add(Dense(units=96, activation='sigmoid', input_dim=dimension))
#model.add(Dropout(dropout))
#model.add(Dense(units=96, activation='sigmoid'))
#model.add(Dense(units=48, activation='sigmoid'))        
#model.add(Dense(units=48, activation='sigmoid'))        
sgd = SGD(lr=best_lr)
model.add(Dense(units=6, activation='linear'))
model.compile(optimizer=sgd,loss='mean_squared_error')

model.fit(X_train, y_train,batch_size=best_bs,epochs=epoch,verbose=False, sample_weight=weights)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


# Organizing predicted kinetic parameters
train_A1 = np.zeros(229)
train_E1 = np.zeros(229)
train_m1 = np.zeros(229)
train_A2 = np.zeros(229)
train_E2 = np.zeros(229)
train_m2 = np.zeros(229)
train_A1pred = np.zeros(229)
train_E1pred = np.zeros(229)
train_m1pred = np.zeros(229)
train_A2pred = np.zeros(229)
train_E2pred = np.zeros(229)
train_m2pred = np.zeros(229)
test_A1 = np.zeros(58)
test_E1 = np.zeros(58)
test_m1 = np.zeros(58)
test_A2 = np.zeros(58)
test_E2 = np.zeros(58)
test_m2 = np.zeros(58)
test_A1pred = np.zeros(58)
test_E1pred = np.zeros(58)
test_m1pred = np.zeros(58)
test_A2pred = np.zeros(58)
test_E2pred = np.zeros(58)
test_m2pred = np.zeros(58)


for i in range(len(train_A1)):
    train_A1[i] = y_train[i,0]
    train_E1[i] = y_train[i,1]
    train_m1[i] = y_train[i,2]
    train_A2[i] = y_train[i,3]
    train_E2[i] = y_train[i,4]
    train_m2[i] = y_train[i,5]
    train_A1pred[i] = y_pred_train[i,0]
    train_E1pred[i] = y_pred_train[i,1]
    train_m1pred[i] = y_pred_train[i,2]
    train_A2pred[i] = y_pred_train[i,3]
    train_E2pred[i] = y_pred_train[i,4]
    train_m2pred[i] = y_pred_train[i,5]
    
for i in range(len(test_A1)):
    test_A1[i] = y_test[i,0]
    test_E1[i] = y_test[i,1]
    test_m1[i] = y_test[i,2]
    test_A2[i] = y_test[i,3]
    test_E2[i] = y_test[i,4]
    test_m2[i] = y_test[i,5]
    test_A1pred[i] = y_pred_test[i,0]
    test_E1pred[i] = y_pred_test[i,1]
    test_m1pred[i] = y_pred_test[i,2]
    test_A2pred[i] = y_pred_test[i,3]
    test_E2pred[i] = y_pred_test[i,4]
    test_m2pred[i] = y_pred_test[i,5]
    

# Calculating NN error in predicting kinetic parameters
MAE_train_A1 = metrics.mean_absolute_error(train_A1, train_A1pred)
MAE_test_A1 = metrics.mean_absolute_error(test_A1, test_A1pred)
MAE_train_E1 = metrics.mean_absolute_error(train_E1, train_E1pred)
MAE_test_E1 = metrics.mean_absolute_error(test_E1, test_E1pred)
MAE_train_m1 = metrics.mean_absolute_error(train_m1, train_m1pred)
MAE_test_m1 = metrics.mean_absolute_error(test_m1, test_m1pred)

MAE_train_A2 = metrics.mean_absolute_error(train_A2, train_A2pred)
MAE_test_A2 = metrics.mean_absolute_error(test_A2, test_A2pred)
MAE_train_E2 = metrics.mean_absolute_error(train_E2, train_E2pred)
MAE_test_E2 = metrics.mean_absolute_error(test_E2, test_E2pred)
MAE_train_m2 = metrics.mean_absolute_error(train_m2, train_m2pred)
MAE_test_m2 = metrics.mean_absolute_error(test_m2, test_m2pred)

# print('MAE train A1: ', MAE_train_A1)
# print('MAE test A1: ', MAE_test_A1)
# print('MAE train E1: ', MAE_train_E1)
# print('MAE test E1: ', MAE_test_E1)
# print('MAE train mn1: ', MAE_train_m1)
# print('MAE test m1: ', MAE_test_m1)

# print('MAE train A2: ', MAE_train_A2)
# print('MAE test A2: ', MAE_test_A2)
# print('MAE train E2: ', MAE_train_E2)
# print('MAE test E2: ', MAE_test_E2)
# print('MAE train m2: ', MAE_train_m2)
# print('MAE test m2: ', MAE_test_m2)


yield_train = np.zeros(229)
yield_test = np.zeros(58)
k1_train = np.zeros(229)
k2_train = np.zeros(229)
k1_test = np.zeros(58)
k2_test = np.zeros(58)
t_train = np.zeros(229)
t_test = np.zeros(58)
H0_train = np.zeros(229)
H0_test = np.zeros(58)
Xf_train = np.zeros(229)
Xf_test = np.zeros(58)

# Calculating yield from predicted kinetic parameters
for i in range(len(yield_train)):
    A1 = train_A1pred[i]
    A2 = train_A2pred[i]
    E1 = train_E1pred[i]
    E2 = train_E2pred[i]
    m1 = train_m1pred[i]
    m2 = train_m2pred[i]
    T = X_train[i,0]
    LSR = X_train[i,1]
    CA = X_train[i,2]
    Fx = X_train[i,3]
    t = X_train[i,12]
    k1_train[i] = getK(A1, E1, CA, m1, T)
    k2_train[i] = getK(A2, E2, CA, m2, T)
    H0_train[i] = (Fx/100)/(LSR+1)*1000
    Xf_train[i] = getX(k1=k1_train[i], k2=k2_train[i], H0=H0_train[i], t=t)
    yield_train[i] = Xf_train[i]*100*LSR/(1000*(Fx/100))

# MAE train calculation from yield and yield calculated from kinetic parameters    
MAEyield_train = metrics.mean_absolute_error(yield_trainset, yield_train) 
print('MAE train yield: ', MAEyield_train)


# Calculating yield from predicted kinetic parameters
for i in range(len(yield_test)):
    A1 = test_A1pred[i]
    A2 = test_A2pred[i]
    E1 = test_E1pred[i]
    E2 = test_E2pred[i]
    m1 = test_m1pred[i]
    m2 = test_m2pred[i]
    T = X_test[i,0]
    LSR = X_test[i,1]
    CA = X_test[i,2]
    Fx = X_test[i,3]
    t = X_test[i,12]
    k1_test[i] = getK(A1, E1, CA, m1, T)
    k2_test[i] = getK(A2, E2, CA, m2, T)
    H0_test[i] = (Fx/100)/(LSR+1)*1000
    Xf_test[i] = getX(k1=k1_test[i], k2=k2_test[i], H0=H0_test[i], t=t)
    yield_test[i] = Xf_test[i]*100*LSR/(1000*(Fx/100))

# MAE test calculation from yield and yield calculated from kinetic parameters     
MAEyield_test = metrics.mean_absolute_error(yield_testset, yield_test) 
print('MAE test yield: ', MAEyield_test)

