# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:20:51 2019

@author: musse
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from collections import defaultdict 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics #for accuracy calculation

x_train = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv', delimiter=',')
y_train = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv', delimiter=',')

x_test = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv', delimiter=',')

##Variables of interest
##['amount_tsh', 'funder', 'installer', 'basin', 'region', 'population', 'publi/meeting', 'scheme_management', 'permit', 'construction_year', 'extraction_type_group', 'payment', 'wuality_group', 'quantity_group', 'source_type', 'waterpoint_type_group'] 

x_train_sel = x_train[['id','amount_tsh', 'permit', 'funder', 'basin', 'public_meeting', 'region', 'payment', 'population', 'scheme_management', 'construction_year', 'extraction_type_group', 'quality_group', 'quantity_group', 'source_type', 'waterpoint_type_group']]

##x_train_sel.info()
##Clean training data
##x_train_sel.info()
##x_train_sel.isnull().sum()

##x_train_sel['funder'].value_counts(dropna=False)
##Okay to remove where 'funder' is null
##Will need to recode values into groups
##x_train_sel['public_meeting'].value_counts(dropna=False)
##Okay to remove where 'public_meeting' is null
##x_train_sel['scheme_management'].value_counts(dropna=False)
##Okay to remove where scheme_management is null
##x_train_sel['permit'].value_counts(dropna=False)
##Okay to remove where permit is null

data = pd.concat([x_train_sel, y_train], axis=1)

data = data.dropna()
y = data.iloc[:,-1]
X = data.iloc[:,1:-2]
##48.6k non-null rows to work with. 

##Bin the years column into categorical variable for before and after 2002
##2002 is when the New Partnership for Africa's Development (NEPAD) and MDGs were adopted
##x_train_sel['construction_year'].value_counts()
def bin(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

cut_points = [2002]
labels = ['pre2002', 'post2002']
X['construction_year'] = bin(X['construction_year'], cut_points, labels)

##Recode "Funder" column and group into main waterpoint funders
main_funders = {'Government Of Tanzania': 'Government of Tanzania', 'World Bank':'World Bank', 'Unicef':'Unicef', 'Germany Republi':'Germany Republi', 'Netherlands':'Netherlands', 'Danida':'Danida', 'Hesawa':'Hesawa'}

X['funder'] = X['funder'].map(main_funders)
X['funder'].fillna('Other', inplace=True)

##Recode 'region' into larger regions of the country along poverty lines
# =============================================================================
# #region_groups = {'Tanga':'East',
#                  'Manyara': 'East',
#                  'Kilimanjaro':'East',
#                  'Dar es Salaam': 'East',
#                  'Arusha': 'East',
#                  'Pwani': 'South',
#                  'Dodoma': 'South',
#                  'Morogoro':'South',
#                  'Lindi':'South',
#                  'Mtwara':'South', 
#                  'Ruvuma':'South', 
#                  'Njombe':'South', 
#                  'Iringa':'South', 
#                  'Mbeya':'South',
#                  'Rukwa':'North',
#                  'Katavi':'North',
#                  'Tabora': 'North',
#                  'Kigoma': 'North',
#                  'Geita': 'North',
#                  'Kagera': 'North',
#                  'Mwanza': 'North',
#                  'Simiyu': 'North',
#                  'Mara': 'North',
#                  'Shinyanga': 'North'}
# =============================================================================
#X['region'] = X['region'].map(region_groups)
#X['region'].fillna('Other', inplace=True)


##Get dummy variables for all categorical columns
X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
##X = X.drop(['scheme_management_None'], axis=1)


########## SPLIT INTO TRAIN AND TEST ####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

##
##Beging Decision Tree Classifier
##
status_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)

##Fit to Decision Tree model
status_tree.fit(X_train, y_train)

status_pred = status_tree.predict(X_test)
status_pred = pd.Series(status_pred)

print("Accuracy:", metrics.accuracy_score(y_test, status_pred))

##Adjust parameters to improve accuracy 

status_tree = DecisionTreeClassifier(criterion='entropy', max_depth=18)

##Fit to Decision Tree model
status_tree.fit(X_train, y_train)

status_pred = status_tree.predict(X_test)
status_pred = pd.Series(status_pred)

print("Accuracy:", metrics.accuracy_score(y_test, status_pred))
##Accuracy at 76%. Not bad! 
status_tree = DecisionTreeClassifier(criterion='entropy', max_depth=13, presort=False, splitter='best')

##Fit to Decision Tree model
status_tree.fit(X_train, y_train)

status_pred = status_tree.predict(X_test)
status_pred = pd.Series(status_pred)

print("Accuracy:", metrics.accuracy_score(y_test, status_pred))


##Lets try RandomForestClassifier 


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees

#Encode the status_group labels 

# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
data = pd.concat([x_train_sel, y_train], axis=1)

data = data.dropna()
data2 = data
le = LabelEncoder()
data2['status_group'] = le.fit_transform(data2['status_group'])
y2 = data2['status_group']

X_train, X_test, y_train2, y_test2 = train_test_split(X, y2, test_size=0.3)

rf = RandomForestRegressor(n_estimators = 82)
# Train the model on training data
rf.fit(X_train, y_train2)
print('R^2 Training Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf.score(X_train, y_train2),
                                                                                             rf.score(X_test, y_test2)))

##Random Forest doesn't perform as well. Simple Decision Tree works fine. 








