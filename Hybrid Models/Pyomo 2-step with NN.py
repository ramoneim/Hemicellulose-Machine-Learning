#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import math
from sklearn import metrics
from scipy import optimize
from pyomo.environ import *
from pyomo.dae import *
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

rho_w = 1000
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


# Creating time and yield list of lists for each experimental set
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

# Creating an array with weights that are proportional to the length of number of datapoints in each experimental set
for i in range(len(TotalT_set)):
    array = TotalT_set[i]
    length = len(array)
    Weights_set[i] = length
    
YieldSol_train = np.zeros(len(CA_set))

# Converting CA and temperature
for i in range(len(CA_set)):
    CA_set[i] = CA_set[i]*98.079/2/10
    if Temp_set[i]>450:
        Temp_set[i]=450
    else: 
        Temp_set[i]=Temp_set[i]
    if Temp_set[i]==0:
        print('There are 0 K temperatures')

kinetic_params = []
MAE_all = []
    
## START OF STEP 1    
    
# Running an optimization loop for each experiment    
for i in range(len(CA_set)):
    print('Index: ', i)
    Time_set = TotalT_set[i]
    Xf_set = np.zeros_like(Time_set)
    Yield_set2 = Yield_set[i]
    CA = CA_set[i]
    T = Temp_set[i]
    LSR = LSR_set[i]
    Fx = Fx_set[i]
    H0 = (Fx/100)/(LSR+1)*1000
    
    X0 = np.zeros_like(Xf_set)
    
    N_sample = len(Xf_set)  
    Sample = range(N_sample) 
    N_Time = len(TotalT_set[i])
    dt = [t/N_Time for t in Time_set]
   
    Time = range(N_Time)                    
    Time_ = range(N_Time-1) 
    print(Time_)
    
    # Starting pyomo concrete model
    model = ConcreteModel()
   
    # Variable definitions
    model.Ah = Var(within=PositiveReals, bounds=(1e8, 1e30))  
    model.Ad = Var(within=PositiveReals, bounds=(1e8, 1e30))  
    model.Eh = Var(within=PositiveReals, bounds=(50, 300))     
    model.Ed = Var(within=PositiveReals, bounds=(50, 300))    
    model.Ah_CA = Var(within=PositiveReals, bounds=(1e8, 1e30))  
    model.Ad_CA = Var(within=PositiveReals, bounds=(1e8, 1e30))  
    model.Eh_CA = Var(within=PositiveReals, bounds=(50, 300))     
    model.Ed_CA = Var(within=PositiveReals, bounds=(50, 300))      
    model.mh_CA = Var(within=PositiveReals,bounds=(0.1,1))         
    model.md_CA = Var(within=PositiveReals,bounds=(0.1,3))        
    
    
    model.kh = Var(Sample, within=PositiveReals,bounds=(1e-20,1e30), initialize=1)        
    model.kd = Var(Sample, within=PositiveReals,bounds=(1e-20,1e30), initialize=1)
    
    model.H = Var(Sample, Time, within=Reals, initialize=0.5)
    model.X = Var(Sample, Time, within=Reals, initialize=0.5)
    model.yield_cal = Var(Sample, within=Reals, initialize=50)
    
    # Creating model constraints
    model.constraints = ConstraintList() 
    
    # Defining Arrhenius relationships
    for s in Sample:
        if np.any(CA==0):
            model.constraints.add((model.kh[s]) == model.Ah*exp(-model.Eh/(R*T) ))
            model.constraints.add((model.kd[s]) == model.Ad*exp(-model.Ed/(R*T)) )
        else:
            model.constraints.add((model.kh[s]) == model.Ah_CA*(CA**(model.mh_CA))*exp(-model.Eh_CA/(R*T) ))
            model.constraints.add((model.kd[s]) == model.Ad_CA*(CA**(model.md_CA))*exp(-model.Ed_CA/(R*T) ))
    
    # Defining system of ODEs for the simple kinetic model
    for s in Sample:
        for t in Time_:
            model.constraints.add( model.H[s,t+1] - model.H[s,t] == -model.kh[s]*model.H[s,t+1]*dt[s] )
            model.constraints.add( model.X[s,t+1] - model.X[s,t]   == (model.kh[s]*model.H[s,t+1] - model.kd[s]*model.X[s,t+1])*dt[s]  ) 
    
    # Initial conditions
    for s in Sample:
        model.constraints.add(model.H[s,0] == (Fx*rho_w/(100*(LSR))))
        model.constraints.add(model.X[s,0] == X0[s])
    
    # Calculating yield
    for s in Sample:
        model.constraints.add(model.yield_cal[s] == 100*model.X[s, N_Time-1]*LSR/(1000*(Fx/100)))
    
    # Objective function
    model.objective = Objective(expr = sum((model.yield_cal[s]-Yield_set2[s])**2 for s in Sample))
    
    solver = SolverFactory('ipopt')
    solver.options['max_iter']=10000
    solver.solve(model, tee=True)
    
    Yield_calctest = np.zeros_like(Time_set)
    
    bestAh = model.Ah.value
    bestAd = model.Ad.value
    bestEh = model.Eh.value
    bestEd = model.Ed.value
    bestAh_CA = model.Ah_CA.value
    bestAd_CA = model.Ad_CA.value
    bestEh_CA = model.Eh_CA.value
    bestEd_CA = model.Ed_CA.value
    bestmh_CA = model.mh_CA.value
    bestmd_CA = model.md_CA.value
    
    
    for s in Sample:
        Yield_calctest[s] = model.yield_cal[s].value
    
    # MAE calculation as well as organization of calculated kinetic parameters from optimization
    MAE = metrics.mean_absolute_error(Yield_set2, Yield_calctest)
    kineticparams = [bestAh, bestAd, bestEh, bestEd, bestAh_CA, bestAd_CA, bestEh_CA, bestEd_CA,bestmh_CA, bestmd_CA]
    print('Kinetic Parameters: ', kineticparams)
    kinetic_params.append(kineticparams)
    MAE_all.append(MAE)
    
print('MAE all: ', MAE_all)
print('Max MAE: ', max(MAE_all))



## START OF STEP 2


# Redefining the kinetic parameters as the 'Y' (output) for the neural network
Y = kinetic_params
Y = np.transpose(Y)
Y = np.transpose(Y)
X_new = np.zeros((287,15))

Y = Y.astype(float)

# Converting any NaN values to 0
for i in range(len(Y)):
    row = Y[i]
    row=np.nan_to_num(row)
    Y[i] = row


feats = range(14)

# Arranging features into an X array and splitting the test and train datasets
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

# Extracting info from the X dataset and redefining for NN
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
model.add(Dropout(dropout))
model.add(Dense(units=96, activation='sigmoid'))
model.add(Dense(units=48, activation='sigmoid'))        
model.add(Dense(units=48, activation='sigmoid'))        
sgd = SGD(lr=best_lr)
model.add(Dense(units=10, activation='linear'))
model.compile(optimizer=sgd,loss='mean_squared_error')

model.fit(X_train, y_train,batch_size=best_bs,epochs=epoch,verbose=False, sample_weight=weights)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Arranging kinetic parameters in their respective arrays
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

# Finding yield from predicted kinetic parameters
for i in range(len(yield_train)):
    T = X_train[i,0]
    LSR = X_train[i,1]
    CA = X_train[i,2]
    Fx = X_train[i,3]
    t = X_train[i,12]  
    if (CA==0):
        A1 = y_pred_train[i,0]
        A2 = y_pred_train[i,1]
        E1 = y_pred_train[i,2]
        E2 = y_pred_train[i,3]
        m1 = 0
        m2 = 0
    else:
        A1 = y_pred_train[i,4]
        A2 = y_pred_train[i,5]
        E1 = y_pred_train[i,6]
        E2 = y_pred_train[i,7]
        m1 = y_pred_train[i,8]
        m2 = y_pred_train[i,9]
    k1_train[i] = getK(A1, E1, CA, m1, T)
    k2_train[i] = getK(A2, E2, CA, m2, T)
    H0_train[i] = (Fx/100)/(LSR+1)*1000
    Xf_train[i] = getX(k1=k1_train[i], k2=k2_train[i], H0=H0_train[i], t=t)
    yield_train[i] = Xf_train[i]*100*LSR/(1000*(Fx/100))

# Calculating MAE train from predicted kinetic parameters and original yield    
MAEyield_train = metrics.mean_absolute_error(yield_trainset, yield_train) 
print('MAE train yield: ', MAEyield_train)


# Finding yield from predicted kinetic parameters
for i in range(len(yield_test)):
    T = X_test[i,0]
    LSR = X_test[i,1]
    CA = X_test[i,2]
    Fx = X_test[i,3]
    t = X_test[i,12]  
    if (CA==0):
        A1 = y_pred_test[i,0]
        A2 = y_pred_test[i,1]
        E1 = y_pred_test[i,2]
        E2 = y_pred_test[i,3]
        m1 = 0
        m2 = 0
    else:
        A1 = y_pred_test[i,4]
        A2 = y_pred_test[i,5]
        E1 = y_pred_test[i,6]
        E2 = y_pred_test[i,7]
        m1 = y_pred_test[i,8]
        m2 = y_pred_test[i,9]
    k1_test[i] = getK(A1, E1, CA, m1, T)
    k2_test[i] = getK(A2, E2, CA, m2, T)
    H0_test[i] = (Fx/100)/(LSR+1)*1000
    Xf_test[i] = getX(k1=k1_test[i], k2=k2_test[i], H0=H0_test[i], t=t)
    yield_test[i] = Xf_test[i]*100*LSR/(1000*(Fx/100))

# Calculating MAE train from predicted kinetic parameters and original yield     
MAEyield_test = metrics.mean_absolute_error(yield_testset, yield_test) 
print('MAE test yield: ', MAEyield_test)

