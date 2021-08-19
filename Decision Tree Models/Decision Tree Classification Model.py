#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:09:32 2020

@author: ranam
"""

import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from graphviz import Source
from six import StringIO
from IPython.display import Image


## Extracting Data from CSV File 
df = pd.read_csv("PreparedDataAll - Copy.csv")
factors = ['Ro', 'logRo', 'P', 'logP', 'H', 'logH']
labels_to_drop_front = [['TotalT'],['Temp'], ['LSR'], ['CA'], ['Size'], ['IsoT'], ['F_X'],]
labels_to_drop_back = [['Ro', 'logRo'],  ['P', 'logP'], ['H', 'logH'], factors  ]
labels_to_drop_all = labels_to_drop_front + labels_to_drop_back
labels_to_drop_front_flat = [item for sublist in labels_to_drop_front for item in sublist]
labels_to_scale = labels_to_drop_front_flat + factors
X_nonscaled = df[labels_to_scale]
finalCols = labels_to_scale
labels_short = [factors]
X_all = pd.concat([X_nonscaled], ignore_index=True,axis=1)
X_all.columns = finalCols

# Looking at distribution of Yield Data
df.hist(column='Yield')


#Performing Decision Tree Analysis on 5 Yield Levels: 0-20, 20-40, 40-60, 60-80, >80
Y2 = (df['Yield Lvls 2'])
df.hist(column='Yield Lvls 2')

#X_train, X_test, y_train2, y_test2 = train_test_split(X_all, Y2, test_size=0.2, random_state=1,stratify=Y2)
X_train, X_test, y_train2, y_test2 = train_test_split(X_all, Y2, test_size=0.2, random_state=1)

clf2=DecisionTreeClassifier(max_depth=6, criterion="gini")
clf2 = clf2.fit(X_train,y_train2)

y_pred2=clf2.predict(X_test)
Accuracy2 = metrics.accuracy_score(y_test2, y_pred2)
MAE2 = metrics.mean_absolute_error(y_test2, y_pred2)

fn2 = finalCols
cn2 = ['Level 1: Yield 0-20', 'Level 2: Yield 20-40', 'Level 3: Yield 40-60', 'Level 4: Yield 60-80', 'Level 5: Yield>80']

plt.figure(figsize=(25,25))
plot_tree(clf2,feature_names=fn2,class_names=cn2, filled=True)
plt.title('Decision Tree on Kinetic Models 2')
plt.show()


# Performing Decision Tree Analysis on 5 Yield Levels: 0-50, 51-70, 71-80, 81-90, >90
Y3 = (df['Yield Lvls 3'])
df.hist(column='Yield Lvls 3')

#X_train, X_test, y_train3, y_test3 = train_test_split(X_all, Y3, test_size=0.2, random_state=1, stratify=Y3) 
X_train, X_test, y_train3, y_test3 = train_test_split(X_all, Y3, test_size=0.2, random_state=1) 

clf3train=DecisionTreeClassifier(max_depth=6, criterion="gini")
clf3 = clf3train.fit( X_train,y_train3)


y_pred3=clf3.predict(X_test)
y_predtrain3 = clf3.predict(X_train)
Accuracy3 = metrics.accuracy_score(y_test3, y_pred3)
Accuracytrain3 = metrics.accuracy_score(y_train3, y_predtrain3)
MAE3_test = metrics.mean_absolute_error(y_test3, y_pred3)
MAE3_train = metrics.mean_absolute_error(y_train3, y_predtrain3)

fn3 = finalCols
cn3 = ['Level 1: Yield 0-50', 'Level 2: Yield 51-70', 'Level 3: Yield 71-80', 'Level 4: Yield 81-90', 'Level 5: Yield>90']
graph = Source(export_graphviz(clf3, feature_names=fn3))
# plt.figure(figsize=(400,400))
# plot_tree(clf3,feature_names=fn3,class_names=cn3, filled=True)
# plt.title('Decision Tree on Kinetic Models 3')
# plt.show()



# Printing Accuracy for all Yield Level DT Analyses
print("Accuracy score % for 5 yield levels: ", 100*Accuracy2)
print("MAE for 5 yield levels: ", MAE2)
print("Accuracy score % for 5 yield levels with defined distribution levels: ", 100*Accuracy3)
print("Accuracy_train score % for 5 yield levels with defined distribution levels: ", 100*Accuracytrain3)
print("MAE for 5 yield levels with defined distribution levels: ", MAE3_test)
print("MAE_train for 5 yield levels with defined distribution levels: ", MAE3_train)




# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.displot(df, x="Yield Lvls 3", discrete=True)
plt.xlabel("Yield Level Class")


# In[ ]:


clf3.tree_.impurity
sns.displot(clf3.tree_.impurity, bins=20)
plt.xlabel("Gini Value")


# In[ ]:


import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


# In[ ]:

# Creating decision tree plot
dotfile = StringIO()
columnnames = ['Total Time', 'Temperature', 'LSR', 'CA', 'Particle Size','Isothermal Time','Fx','Ro','logRo', 'P', 'logP','H', 'logH' ]
classnames = ['Level 1: Yield 0-50', 'Level 2: Yield 51-70', 'Level 3: Yield 71-80', 'Level 4: Yield 81-90', 'Level 5: Yield 91-100'] 
export_graphviz(clf3train, out_file=dotfile, filled=True, rounded=True, special_characters=True, feature_names=columnnames, class_names=classnames)
graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png('DecisionTree.png')

