# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:20:51 2019

@author: musse
"""

#Import common modules.
import pandas as pd
import numpy as np
import re
import os
import random

##Load data
##This will depend on where you saved the data files
os.chdir('C:\\...')

data = pd.read_csv('training_variables.csv', delimiter=',', index_col='id')
train_y = pd.read_csv('training_labels.csv', delimiter=',')
test_y = pd.read_csv('test_variables.csv', delimiter=',')

##Data cleaning and mining
data.info()

data = data.dropna(how='any')
data = data[['amount_tsh', 'permit', 'funder', 'basin', 'public_meeting', 'region', 'payment', 'population', 'scheme_management', 'construction_year', 'extraction_type_group', 'quality_group', 'quantity_group', 'source_type', 'waterpoint_type_group']]

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

##Let'try the Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
status_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)

##Fit to Decision Tree model
status_tree.fit(X_train, y_train)

status_pred = status_tree.predict(X_test)
status_pred = pd.Series(status_pred)

##Measure model accuracy 
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, status_pred))
##94.5%.... Not bad! 

##Adjust parameters to improve accuracy 

status_tree = DecisionTreeClassifier(criterion='entropy', max_depth=18)

##Fit to Decision Tree model
status_tree.fit(X_train, y_train)

status_pred = status_tree.predict(X_test)
status_pred = pd.Series(status_pred)

print("Accuracy:", accuracy_score(y_test, status_pred))
##Accuracy at 96%.1.... Nice bump up!

##Naturally, it's worth trying a Random Forest Classifer

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees


rf = RandomForestClassifier(n_estimators=1000, max_depth=250, min_samples_leaf=50, max_features=.2, n_jobs=-1 )
# Train the model on training data
rf.fit(X_train, y_train)
status_pred2  = rf.predict(X_test)
status_pred2 = pd.Series(status_pred2)
print("Accuracy:", accuracy_score(y_test, status_pred2))
##94.8%. Not bad, but no real advantage compared to decision tree. 


##We could try SVM or tune the RF Classifier using grid_search to slightly improve our predictions.
##Based on how much resources this basic classifier required from my machine, I'll settle for these results. 








