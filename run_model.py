#!/usr/bin/env python
# coding: utf-8

# generak
import pandas as pd
import numpy as np
from pandas import to_datetime


# plot libarys
import seaborn as sns
import matplotlib.pyplot as plt

# Model preperation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
 

# Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# Model Metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, plot_confusion_matrix

# for merging the dataframes
import os, glob
import json

# further libarys
import itertools
from sklearn.tree import export_graphviz

# saveing
import pickle

# handling terminal arguments
import sys

'''
Python Skript, which takes a train, test data and model as argumend 
and saves the modelprediction and the classification repotr
also it will print the report and the confisuion matrix, the latter it also will plot in a new window
'''

#import pickle
def load_data(test, train, model):
    with open(test, 'rb') as f:
        X_test, y_test = pickle.load(f)

    with open(train, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    model = pickle.load(open(model, 'rb'))

    return X_train, y_train, X_test, y_test, model
    



def prediction(X_train, X_test, y_train, y_test, model):
    # Training predictions 
    y_train_pred = model.predict(X_train)
    
    # Testing predictions, to determine performance
    y_pred = model.predict(X_test)

    print(f"--- MODEL PARAMETERS {'-'*10}")
    print(json.dumps(model.get_params(), indent=4))
    print(f"--- CLASSIFICATION REPORT {'-'*10}")
    print(classification_report(y_test,y_pred))
    print(f"--- CONFUSION MATRIX {'-'*10}")
    print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    return classification_report(y_test,y_pred), y_train_pred, y_pred
    
    
### Call funktion etc
test_file = sys.argv[1]
train_file = sys.argv[2]
model_file = sys.argv[3]
X_train, y_train, X_test, y_test, model =  load_data(test_file, train_file, model_file)


report, y_train_pred, y_pred = prediction(X_train, X_test, y_train, y_test, model)

## saves
with open('prediction.csv', 'wb') as f:
    pickle.dump([y_train_pred, y_pred], f)


with open('class_report.csv', 'wb') as f:
    pickle.dump([report],f)

## python pred_print_plot_test.py 'test_test.csv' 'train_test.csv' 'model_test.csv' 