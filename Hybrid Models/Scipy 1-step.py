# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:27:54 2021

@author: ranam
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:14:38 2021

@author: ranam
"""


from sklearn import metrics
import pandas as pd
import numpy as np
import math
import collections
from scipy import optimize
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from numpy import concatenate



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


Y = (df['Yield'])
y_train = Y.tolist()
YIELD_train = y_train


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

# Converting acid concentration and temperature
for i in range(len(CA_train)):
    CA_train[i]=CA_train[i]*98.079/2/10
    if Temp_train[i]>450:
        Temp_train[i]=450
    else:
        Temp_train[i]=Temp_train[i]

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
#Time_set = np.zeros(len(rowS))

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


feats = range(14)
X_new = np.zeros((287,14))
Y = np.zeros(287)

# Defining X and Y for the model
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
    #X_new[i,14]=YIELD_train[index]
    Y[i]=YIELD_train[index]

# Converting all X values to float    
X_newnew = []
X_new = np.array(X_new, dtype=float) #  convert using numpy
for i in range(len(X_new)):
    row = X_new[i,:]
    rownew = [float(i) for i in row] 
    X_newnew.append(rownew)


# Converting all Y values to float     
Yield_set_new = []
for i in range(len(Yield_set)):
    rowyield = Yield_set[i]
    rowyield = [float(i) for i in rowyield]
    Yield_set_new.append(rowyield)
    
Yield_set = Yield_set_new

Y = np.array(Y, dtype=float) #  convert using numpy
Y = [float(i) for i in Y] 
    
# Splitting into training and testing sets    
X_train, X_test, y_train, y_test = train_test_split(X_newnew, Y, test_size=0.2, random_state=None, shuffle=False)

Yield_set = np.transpose(np.transpose(Yield_set))

# Function for flattening arrays or list of lists
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

# Defining all kinetic functions
def getK (A, E, Ca, m, T):
    # A in min^(-1); E in kJ/mol, Ca in %, and T in K
    if T>450:
        T=450
    if Ca>0: 
        k = A*Ca**m*math.exp(-E/(8.3143e-3*T))
        return k
    else:
        k = A*math.exp(-E/(8.3143e-3*T))
        return k

def getKnew(A, E, m, i):
    T = X_train[i][0]
    CA = X_train[i][2]
    if T>450:
        T=450
    if CA>0: 
        k = A*CA**m*math.exp(-E/(8.3143e-3*T))
        return k
    else:
        k = A*math.exp(-E/(8.3143e-3*T))
        return k


def getX(k1,k2, H0, t):
    num = -k1*H0*(math.exp(-k1*t)-math.exp(-k2*t))
    denom = k1 - k2
    return num/denom

# Defining ODEs for SOLVE IVP 
def dxdt(t, y, i, index_params, params):
    # h, x = y
    A1, E1, m1, A2, E2, m2 = getkinparams(index_params, params)
    k1 = getKnew(A1, E1, m1, i)
    k2 = getKnew(A2, E2, m2, i)
    dhdt = -k1*y[0]
    dxdt = k1*y[0]- k2*y[1]
    return [dhdt, dxdt]

def getkinparams(index_params, params):
    A1 = params[index_params]
    E1 = params[index_params+1]
    m1 = params[index_params+2]
    A2 = params[index_params+3]
    E2 = params[index_params+4]
    m2 = params[index_params+5]
    return (A1, E1, m1, A2, E2, m2)


# Defining ODEs for SOLVE IVP - dif version 
def dxdtnew(t, y, CA, T, A1, A2, m1, m2, E1, E2):
    if CA>0: 
        k1 = A1*CA**m1*math.exp(-E1/(8.3143e-3*T))
        k2 = A2*CA**m2*math.exp(-E2/(8.3143e-3*T))
    else:
        k1 = A1*math.exp(-E1/(8.3143e-3*T))
        k2 = A2*math.exp(-E2/(8.3143e-3*T))
    dhdt = -k1*y[0]
    dxdt = k1*y[0]- k2*y[1]
    return [dhdt, dxdt]


# Defining function that gives yield error 
def get_error(params):
    y_train_pred = []
    for i in range(len(X_train)):
        T = X_train[i][0]
        LSR = X_train[i][1]
        CA = X_train[i][2]
        Fx = X_train[i][3]
        H0 = (Fx/100)/(LSR+1)*1000
        y_set = np.zeros_like(Yield_set[i])
        y_set = Yield_set[i]
        Yield_solset = np.zeros_like(y_set)
        Time_set = TotalT_set[i]
        print('Index: ', i)
        # Extracting all linear parameters
        x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, z1, z2, z3, z4, z5, w1, w2, w3, w4, w5, l1, l2 , l3, l4, l5, o1, o2, o3, o4, o5 = params[1374:]
        index_params  = i*6
        for j in range(len(y_set)):
            A1 = params[index_params]
            E1 = params[index_params+1]
            m1 = params[index_params+2]
            A2 = params[index_params+3]
            E2 = params[index_params+4]
            m2 = params[index_params+5]
            # Defining linear relationship between features and kinetic parameters
            A1 = x1*T+x2*LSR+x3*CA+x4*Fx+x5
            E1 = y1*T+y2*LSR+y3*CA+y4*Fx+y5
            m1 = z1*T+z2*LSR+z3*CA+z4*Fx+z5
            A2 = w1*T+w2*LSR+w3*CA+w4*Fx+w5
            E2 = l1*T+l2*LSR+l3*CA+l4*Fx+l5
            m2 = o1*T+o2*LSR+o3*CA+o4*Fx+o5
            t = Time_set[j]
        # Defining solve ivp parameters    
        t_0 = min(Time_set)-5
        t_max = max(Time_set)+5
        t_span = [t_0, t_max]   
        init_vals = [H0, 0]
        t_eval = Time_set
        # Integrating the ODE to solve
        sol = solve_ivp(dxdt, t_span, init_vals, t_eval = t_eval, args=(i, index_params, params), dense_output=True)
        # Calculating yield using calculated Xf concentration
        for j in range(len(y_set)):
            Yield_solset[j] = 100*sol.y[1][j]*LSR/(1000*(Fx/100))
        y_train_pred.append(Yield_solset)
        # print('y_train_pred: ', y_train_pred)
    # Arranging and flattening y's and calculating MAE
    y_train_real = []
    for s in range(len(y_train_pred)):
        yrow = Yield_set[s]
        y_train_real.append(yrow)
    y_train_pred = flatten(y_train_pred)
    y_train_real = flatten(y_train_real)
    # print('y pred, y real: ', y_train_pred, y_train_real)
    error = metrics.mean_absolute_error(y_train_real, y_train_pred)
    print('Error: ', error)
    print('Params: ', params)
    return error

kinetic_params = []
errorsab = [] 

# Defining and organizing initial guess for minimization problem 
initialGuessparams = [4.67e16, 142.58, 1.75, 6.51e16, 155.36,1]
variablesguess = np.ones(30)
initialguess = []
for i in range(len(X_train)):
    initialguess.append(initialGuessparams)
initialguess = np.transpose(np.transpose(initialguess))
initialguess=initialguess.flatten()

# Bounds for features if using minimize instead of fmin -- both okay
lowerbound = [1e8, 50, 0.1, 1e8, 50, 0.1]
upperbound = [1e30, 300, 1, 130, 300, 3]
lbvs = -((np.ones(30))*np.inf)
ubvs = (np.ones(30))*np.inf

lbvs = lbvs.reshape(lbvs.shape[0],1)
ubvs = ubvs.reshape(ubvs.shape[0],1)

ub = []
lb = []

for i in range(len(X_train)):
    ub.append(upperbound)
    lb.append(lowerbound)

ub = np.transpose(np.transpose(ub))
lb = np.transpose(np.transpose(lb))
ub = ub.flatten()
lb = lb.flatten()
ub = ub.reshape(ub.shape[0],1)
ub = concatenate((ub,ubvs), axis=0)
lb = lb.reshape(lb.shape[0],1)
lb = concatenate((lb,lbvs), axis=0)

bounds = concatenate((lb,ub), axis=-1)

# Defining scipy optimization function
initialguess = concatenate((initialguess, variablesguess), axis=0)
initialguess = np.array(initialguess, dtype=float) #  convert using numpy
intialguess = [float(i) for i in initialguess] 
#output = optimize.minimize(get_error, initialguess, bounds=bounds)
output = optimize.fmin(get_error, initialguess,maxiter=10000, full_output=1)
minimum = output[0]
errors = output[1]

print('minimum: ', minimum)
print('errors: ', errors)





