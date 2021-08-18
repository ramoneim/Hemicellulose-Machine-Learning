#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('DataWithAcid.csv')
df['t_pseudo'] = pd.to_numeric(df['t_pseudo'],errors = 'coerce')

Temperature = df.iloc[:,1].values
LSR = df.iloc[:,2].values
CA = df.iloc[:,3].values
Size = df.iloc[:,4].values
IsoTime = df.iloc[:,5].values
HeatingTime = df.iloc[:, 6].values
t_rxn = df.iloc[:,8].values
t_pseudo = df.iloc[:,11].values
F_X = df.iloc[:,12].values
Yield = df.iloc[:,20].values
t_total = np.zeros_like(t_rxn)

for i in range(len(IsoTime)):
    t_total[i] = IsoTime[i] + HeatingTime[i]
    
Size = Size.reshape((Size.shape[0], 1))
LSR = LSR.reshape((LSR.shape[0], 1))
CA = CA.reshape((CA.shape[0], 1))
F_X = F_X.reshape((F_X.shape[0],1))
Temp = Temperature.reshape((Temperature.shape[0], 1))
t_total = t_total.reshape((IsoTime.shape[0], 1))
X = np.concatenate((Size, LSR, CA, F_X, Temp, t_total), axis=1) 
Y = Yield
print("Array type: ", X.dtype)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=1)
print("Train X type: ", train_X.dtype)
print("Train Y type: ", train_Y.dtype)
print("Train X length: ", len(train_X))
print("Test X length: ", len(test_X))


Time_meas = train_X[:,5]
Time_test = test_X[:,5]
Tem_meas = train_X[:,4]
Tem_test = test_X[:,4]
Yield_meas = train_Y
Yield_test = test_Y
Fx_meas = train_X[:,3]
Fx_test = test_X[:,3]
CA_meas = train_X[:,2]
CA_test = test_X[:,2]
X0_meas = np.zeros_like(CA_meas)
X0_test = np.zeros_like(CA_test)
LSR_meas = train_X[:,1]
LSR_test = test_X[:,1]
CAweight_meas = np.zeros_like(CA_meas)
CAweight_test = np.zeros_like(CA_test)

R = 8.314
rho_w = 1000


# In[9]:


from pyomo.environ import *
from pyomo.dae import *
import math
import pandas as pd 
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 


# Discretizing Time for Euler Model
N_sample =  len(Time_meas)
Sample = range(N_sample)
N_Time = 20                        # Of time steps
dt =  [t/N_Time for t in Time_meas] # step

N_sample_test =  len(Time_test)
Sample_test = range(N_sample_test)
N_Time_test = 20                        # Of time steps
dt_test =  [t/N_Time_test for t in Time_test] # step

# Temperature and Acid Concentration Conversion    
for i in range(len(Tem_meas)):
    if Tem_meas[i]>450:
        Tem_meas[i]=450
    if Tem_meas[i] == 0:
        print('There are 0 K temperatures')
        
for i in range(len(CA_meas)):
    CAweight_meas[i]=CA_meas[i]*98.079/2/10

for i in range(len(Tem_test)):
    if Tem_test[i]>450:
        Tem_test[i]=450
    if Tem_test[i] == 0:
        print('There are 0 K temperatures')
        
for i in range(len(CA_test)):
    CAweight_test[i]=CA_test[i]*98.079/2/10
    
Time = range(N_Time)                    # 
Time_ = range(N_Time-1) 
print(Time_)
Time_test2 = range(N_Time_test)                    # 
Time__test = range(N_Time_test-1) 
print(Time__test)

# Defining Pyomo model variables
model = ConcreteModel()
# model.logAf = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #1,8e14
# model.logAs = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #2.3e11
# model.logAd = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #1.4e14
# model.logAo = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #1.4e14
# model.Ef = Var(within=PositiveReals, bounds=(100, 1e8), initialize = 1e5)     #1.4e5
# model.Es = Var(within=PositiveReals, bounds=(100, 1e8), initialize = 1e5)     #8.6e4
# model.Ed = Var(within=PositiveReals, bounds=(100, 1e8), initialize = 1e5)     #1.4e5
# model.Eo = Var(within=PositiveReals, bounds=(100, 1e8), initialize = 1e5)     #1.4e5
model.alpha = Var(within=PositiveReals, bounds=(0.60,0.77), initialize = 0.7)     #0.36

# model.logAf_CA = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #1,8e14
# model.logAs_CA = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #2.3e11
# model.logAd_CA = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #1.4e14
# model.logAo_CA = Var(within=PositiveReals, bounds=(20, 70), initialize = 32)  #1.4e14
model.Af_CA = Var(within=PositiveReals, bounds=(1e10, 1e30), initialize=8e14)
model.As_CA = Var(within=PositiveReals, bounds=(1e10, 1e30), initialize=8e14)
model.Ad_CA = Var(within=PositiveReals, bounds=(1e10, 1e30), initialize=8e14)
model.Ao_CA = Var(within=PositiveReals, bounds=(1e10, 1e30), initialize=8e14)
model.Ef_CA = Var(within=PositiveReals, bounds=(1000, 1e8), initialize = 1e5)     #1.4e5
model.Es_CA = Var(within=PositiveReals, bounds=(1000, 1e8), initialize = 1e5)     #8.6e4
model.Ed_CA = Var(within=PositiveReals, bounds=(1000, 1e8), initialize = 1e5)     #1.4e5  
model.Eo_CA = Var(within=PositiveReals, bounds=(1000, 1e8), initialize = 1e5)     #1.4e5  
model.mf_CA = Var(within=PositiveReals, bounds=(1.0,3.8), initialize = 1)         #5.5
model.ms_CA = Var(within=PositiveReals, bounds=(1.0,3.8), initialize = 1)         #3.8 
model.md_CA = Var(within=PositiveReals, bounds=(0.6,3.8), initialize = 1)         #1.3
model.mo_CA = Var(within=PositiveReals, bounds=(1.1,3.8), initialize = 2)

model.kf = Var(Sample, within=PositiveReals,bounds=(1e-10,1000000), initialize=1)         #1e-9 -  1e-3
model.ks = Var(Sample, within=PositiveReals,bounds=(1e-10,1000000), initialize=1)         #
model.kd = Var(Sample, within=PositiveReals,bounds=(1e-10,1000000), initialize=1)
model.ko = Var(Sample, within=PositiveReals,bounds=(1e-10,1000000), initialize=1)

model.Hf = Var(Sample, Time, within=Reals, initialize=0.5)
model.Hs = Var(Sample, Time, within=Reals, initialize=0.5)
model.X = Var(Sample, Time, within=Reals, initialize=0.5)
model.O = Var(Sample, Time, within=Reals, initialize=0.5)
model.yield_cal = Var(Sample, within=Reals, initialize=50)

model.constraints = ConstraintList()

# Arrhenius Model
for s in Sample:
    if np.any(CA_meas[s]==0):
        print('The training dataset contsins no acid datapoints')
#         model.constraints.add(log(model.kf[s]) == model.logAf - model.Ef/(R*Tem_meas[s]) )
#         model.constraints.add(log(model.ks[s]) == model.logAs - model.Es/(R*Tem_meas[s]) ) 
#         model.constraints.add(log(model.kd[s]) == model.logAd - model.Ed/(R*Tem_meas[s]) )
#         model.constraints.add(log(model.ko[s]) == model.logAo - model.Eo/(R*Tem_meas[s]) ) 
    else:
        model.constraints.add(log(model.kf[s]) == log(model.Af_CA) + model.mf_CA*log(CAweight_meas[s]) - model.Ef_CA/(R*Tem_meas[s]) )
        model.constraints.add(log(model.ks[s]) == log(model.As_CA) + model.ms_CA*log(CAweight_meas[s]) - model.Es_CA/(R*Tem_meas[s]) )
        model.constraints.add(log(model.kd[s]) == log(model.Ad_CA) + model.md_CA*log(CAweight_meas[s]) - model.Ed_CA/(R*Tem_meas[s]) )
        model.constraints.add(log(model.ko[s]) == log(model.Ao_CA) + model.mo_CA*log(CAweight_meas[s]) - model.Eo_CA/(R*Tem_meas[s]) )

# Discretize Using Implicit Euler
for s in Sample:
    for t in Time_:
        model.constraints.add( model.Hf[s,t+1] - model.Hf[s,t] == -model.kf[s]*model.Hf[s,t+1]*dt[s] ) 
        model.constraints.add( model.Hs[s,t+1] - model.Hs[s,t] == -model.ks[s]*model.Hs[s,t+1]*dt[s] )
        model.constraints.add( model.X[s,t+1] - model.X[s,t]   == (model.ko[s]*model.O[s,t+1]- model.kd[s]*model.X[s,t+1])*dt[s]  )
        model.constraints.add( model.O[s,t+1] - model.O[s,t] == (model.kf[s]*model.Hf[s,t+1]+model.ks[s]*model.Hs[s,t+1] -model.ko[s]*model.O[s,t+1])*dt[s])
        
#Initial Values
for s in Sample:
    model.constraints.add(model.Hf[s,0] == model.alpha*(Fx_meas[s]*rho_w/(100*(LSR_meas[s]+1))))        ###double check the formulation
    model.constraints.add(model.Hs[s,0] == (1-model.alpha)*(Fx_meas[s]*rho_w/(100*(LSR_meas[s]+1))))
    model.constraints.add(model.X[s,0] == X0_meas[s])
    model.constraints.add(model.O[s,0] == X0_meas[s])

    
    
#Yield Calculation
for s in Sample:
    model.constraints.add(model.yield_cal[s] == 100*(model.X[s, N_Time-1]+model.O[s,N_Time-1])*LSR_meas[s]/(rho_w*(Fx_meas[s]/100)))
    
#--------------------------------------- Calculation of MAE for Test Data ---------------------------------

# Objective function
model.objective = Objective(expr = sum((model.yield_cal[s]-Yield_meas[s])**2 for s in Sample))


solver=SolverFactory('ipopt')
solver.options['max_iter'] = 10000
# solve
solver.solve(model,tee=True)

# bestlogAf = model.logAf.value
# bestlogAs = model.logAs.value
# bestlogAd = model.logAd.value
# bestlogAo = model.logAo.value
# bestEf = model.Ef.value
# bestEs = model.Es.value
# bestEd = model.Ed.value
# bestEo = model.Eo.value
bestalpha = model.alpha.value


# bestlogAf_CA = model.logAf_CA.value
# bestlogAs_CA = model.logAs_CA.value
# bestlogAd_CA = model.logAd_CA.value
# bestlogAo_CA = model.logAo_CA.value
bestAf_CA = model.Af_CA.value
bestAs_CA = model.As_CA.value
bestAd_CA = model.Ad_CA.value
bestAo_CA = model.Ao_CA.value
bestEf_CA = model.Ef_CA.value
bestEs_CA = model.Es_CA.value
bestEd_CA = model.Ed_CA.value
bestEo_CA = model.Ed_CA.value
bestmf_CA = model.mf_CA.value
bestms_CA = model.ms_CA.value
bestmd_CA = model.md_CA.value
bestmo_CA = model.md_CA.value

# print('logAf: ',model.logAf.value)
# print('logAs: ',model.logAs.value)
# print('logAd: ',model.logAd.value)
# print('logAo: ',model.logAo.value)
# print('Ef (J/mol-K): ',model.Ef.value)
# print('Es (J/mol-K): ',model.Es.value)
# print('Ed (J/mol-K): ',model.Ed.value)
# print('Ei (J/mol-K): ',model.Eo.value)
print('Alpha: ',model.alpha.value)


# print('logAf_CA: ',model.logAf_CA.value)
# print('logAs_CA: ',model.logAs_CA.value)
# print('logAd_CA: ',model.logAd_CA.value)
# print('logAo_CA: ',model.logAo_CA.value)
print('Af_CA: ',model.Af_CA.value)
print('As_CA: ',model.As_CA.value)
print('Ad_CA: ',model.Ad_CA.value)
print('Ao_CA: ',model.Ao_CA.value)
print('Ef_CA (J/mol-K): ',model.Ef_CA.value)
print('Es_CA (J/mol-K): ',model.Es_CA.value)
print('Ed_CA (J/mol-K): ',model.Ed_CA.value)
print('Eo_CA (J/mol-K): ',model.Eo_CA.value)
print('mf: ',model.mf_CA.value)
print('ms: ',model.ms_CA.value)
print('md: ',model.md_CA.value)
print('mo: ',model.mo_CA.value)

# Plotting datapoint model behavior
Xvalues = np.empty(N_Time)
Hfvalues = np.empty(N_Time)
Hsvalues = np.empty(N_Time)
Ovalues = np.empty(N_Time)
Timepts  = np.linspace(0,Time_meas[6], N_Time)

for t in Time: 
    Xvalues[t] = model.X[6,t].value 
    Hfvalues[t] = model.Hf[6,t].value
    Hsvalues[t] = model.Hs[6,t].value
    Ovalues[t] = model.O[6,t].value
    
plt.plot(Timepts,Xvalues, linestyle='-', label='Xylose Concentration')
plt.plot(Timepts, Hfvalues, linestyle='--', label='Fast-acting Hemicellulose')
plt.plot(Timepts, Hsvalues, linestyle='-.', label='Slow-acting Hemicellulose')
plt.plot(Timepts, Ovalues, linestyle=':', label ='Intermediate Oligomers')
plt.legend()
plt.show()

# print("Time points: ", Timepts)

# print("Hf:  ")
# for t in Time:
#     print(model.Hf[6,t].value)
    
# print("Hs:  ")
# for t in Time:
#     print(model.Hs[6,t].value)

# print("O:  ")
# for t in Time:
#     print(model.O[6,t].value)
    
# print("X:  ")
# for t in Time:
#     print(model.X[6,t].value)

Yield_Calc = np.zeros_like(Time_meas)
print("Predicted Yield:")
# print solutions
for s in Sample:
#     print(model.yield_cal[s].value)
    Yield_Calc[s] = model.yield_cal[s].value

MAE_train = metrics.mean_absolute_error(Yield_meas,Yield_Calc)
# print("MAE_train: ", MAE_train)



model_test = ConcreteModel()
model_test.kf_test = Var(Sample_test, within=PositiveReals)
model_test.ks_test = Var(Sample_test, within=PositiveReals)
model_test.kd_test = Var(Sample_test, within=PositiveReals)
model_test.ko_test = Var(Sample_test, within=PositiveReals)
model_test.Hf_test = Var(Sample_test, Time_test2, within=Reals)
model_test.Hs_test = Var(Sample_test, Time_test2, within=Reals)
model_test.X_test = Var(Sample_test, Time_test2, within=Reals)
model_test.O_test = Var(Sample_test, Time_test2, within=Reals)
model_test.yield_cal_test = Var(Sample_test, within=Reals)

model_test.constraints = ConstraintList()
for s in Sample_test:
    if np.any(CA_test[s]==0):
        print('There are no acid datapoints in the testing dataset')
#         model_test.constraints.add(log(model_test.kf_test[s]) == bestlogAf - bestEf/(R*Tem_test[s]) )
#         model_test.constraints.add(log(model_test.ks_test[s]) == bestlogAs - bestEs/(R*Tem_test[s]) )
#         model_test.constraints.add(log(model_test.kd_test[s]) == bestlogAd - bestEd/(R*Tem_test[s]) )
#         model_test.constraints.add(log(model_test.ko_test[s]) == bestlogAo - bestEo/(R*Tem_test[s]) )
    else:
        model_test.constraints.add(log(model_test.kf_test[s]) == log(bestAf_CA) + bestmf_CA*log(CAweight_test[s]) - bestEf_CA/(R*Tem_test[s]) )
        model_test.constraints.add(log(model_test.ks_test[s]) == log(bestAs_CA) + bestms_CA*log(CAweight_test[s]) - bestEs_CA/(R*Tem_test[s]) )
        model_test.constraints.add(log(model_test.kd_test[s]) == log(bestAd_CA) + bestmd_CA*log(CAweight_test[s]) - bestEd_CA/(R*Tem_test[s]) )
        model_test.constraints.add(log(model_test.ko_test[s]) == log(bestAo_CA) + bestmo_CA*log(CAweight_test[s]) - bestEo_CA/(R*Tem_test[s]) )


for s in Sample_test:
    for t in Time__test:
        model_test.constraints.add((model_test.Hf_test[s,t+1] - model_test.Hf_test[s,t]) == -model_test.kf_test[s]*model_test.Hf_test[s,t+1]*dt_test[s])
        model_test.constraints.add((model_test.Hs_test[s,t+1] - model_test.Hs_test[s,t]) == -model_test.ks_test[s]*model_test.Hs_test[s,t+1]*dt_test[s])
        model_test.constraints.add((model_test.X_test[s,t+1] - model_test.X_test[s,t]) == (model_test.ko_test[s]*model_test.O_test[s,t+1] -model_test.kd_test[s]*model_test.X_test[s,t+1])*dt_test[s])
        model_test.constraints.add((model_test.O_test[s,t+1] - model_test.O_test[s,t]) == (model_test.kf_test[s]*model_test.Hf_test[s,t+1]+model_test.ks_test[s]*model_test.Hs_test[s,t+1] - model_test.ko_test[s]*model_test.O_test[s,t+1])*dt_test[s])
        
        
#Initial Values for Test Set                                                                                                                                                                          
for s in Sample_test:
    model_test.constraints.add(model_test.Hf_test[s,0] == bestalpha*(Fx_test[s]*rho_w/(100*(LSR_test[s]+1))))
    model_test.constraints.add(model_test.Hs_test[s,0] == (1-bestalpha)*(Fx_test[s]*rho_w/(100*(LSR_test[s]+1))))
    model_test.constraints.add(model_test.X_test[s,0] == X0_test[s])
    model_test.constraints.add(model_test.O_test[s,0] == X0_test[s])

#Yield Calculation for Test Set                                                                                                                                                                       
for s in Sample_test:
    model_test.constraints.add(model_test.yield_cal_test[s] == 100*(model_test.X_test[s, N_Time_test-1]+model_test.O_test[s,N_Time_test-1])*LSR_test[s]/(rho_w*(Fx_test[s]/100)))

# Solving testing data
solver=SolverFactory('ipopt')
solver.solve(model_test,tee=True)


Yield_CalcTest = np.zeros_like(Time_test)
for s in Sample_test:
    Yield_CalcTest[s] = model_test.yield_cal_test[s].value
MAE_test = metrics.mean_absolute_error(Yield_test,Yield_CalcTest)
print("MAE_train: ", MAE_train)
print("MAE_test: ", MAE_test)


# In[ ]:




