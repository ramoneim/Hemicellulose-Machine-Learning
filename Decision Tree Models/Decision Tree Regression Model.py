#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:08:17 2021

@author: ranam
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.tree import export_graphviz 
from graphviz import Source
from six import StringIO
import pydotplus
from IPython.display import Image

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
Y = Y.tolist()
#YIELD_train = y_train
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

X = np.zeros((1748,10))

for i in range(len(X)):
    X[i,0]=Temp_train[i]
    X[i,1]=CA_train[i]
    X[i,2]=F_X_train[i]
    X[i,3]=Size_train[i]
    X[i,4]=Ro_train[i]
    #X[i,5]=logRo_train[i]
    X[i,5]=P_train[i]
    #X[i,7]=logP_train[i]
    X[i,6]=H_train[i]
    #X[i,7]=logH_train[i]
    X[i,7]=IsoT_train[i]
    X[i,8]=TotalT_train[i]
    X[i,9]=LSR_train[i]
    

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

regressor = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
regressor.fit(X_train,y_train)

ytrain_pred = regressor.predict(X_train)
ytest_pred = regressor.predict(X_test)

MAE_train = metrics.mean_absolute_error(y_train, ytrain_pred) 
MAE_test = metrics.mean_absolute_error(y_test, ytest_pred) 

print('MAE train: ', MAE_train)
print('MAE test: ', MAE_test)

#export_graphviz(regressor, out_file ='tree.dot',feature_names =['T', 'CA', 'Fx', 'Size', 'Ro', 'logRo', 'P', 'logP', 'H', 'logH', 'IsoT', 'TotalT', 'LSR'])
export_graphviz(regressor, out_file ='treeregressor.dot',feature_names =['T', 'CA', 'Fx', 'Size', 'Ro', 'P', 'H', 'IsoT', 'TotalT', 'LSR']) 


# In[2]:


import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


# In[3]:


dotfile = StringIO()
# columnnames = ['Total Time', 'Temperature', 'LSR', 'CA', 'Particle Size','Isothermal Time','Fx','Ro','logRo', 'P', 'logP','H', 'logH' ]
feature_names =['T', 'CA', 'Fx', 'Size', 'Ro', 'P', 'H', 'IsoT', 'TotalT', 'LSR']
export_graphviz(regressor, out_file=dotfile, filled=True, rounded=True, special_characters=True, feature_names=feature_names)
graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png('DecisionTreeRegressorNew.png')


# In[ ]:




