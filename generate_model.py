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
Python Skript, which takes a dataframe as argumend and saves 
the model and the variables from train_test_split

Includes the whole data preperation with data cleaning and feature engineering.
For further information please see the representing notebook
'''

def load_data(filename):
    #pd.read_pickle('data/Kickstarter.csv')
    #data = pd.read_csv(filename)
    data = pd.read_csv(filename, index_col = [0])
    #data =  pickle.load(open(filename, 'rb'))
    return data
    

## Data cleaning
# Droping columns
def drop_columns(data, col):
    for i in col:
        data.drop(columns = i, inplace=True)
    return data

# Calculate the dates
def calc_datetime(data, columns):
    for i in columns:
        data[i] = pd.to_datetime(data[i], unit = 's')
    return data

# Convert obj to category
def obj_to_categorical(data, catrgorical):
    data[categorical] = data[categorical].astype("category")
    return data

## Feature engineering

# Extract data in dictionary in category column into separate columns with leading `"category_"`.
def ext_dict(data, column):
    data = data.join(pd.DataFrame(data[column].apply(lambda x: json.loads(x)).to_list()).add_prefix(f"category_"))
    data.drop(columns=[column], inplace=True)
    category_out = ["category_id", "category_color", "category_position", "category_urls"]
    data.drop(columns=category_out, inplace=True)
    category_categorical = ["category_parent_id", "category_name", "category_slug"]
    data[category_categorical] = data[category_categorical].astype("category")

    df_cat = data.category_slug.str.title().str.split("/", expand=True).rename(columns={0: "parent_category_name", 1: "subcategory_name"})
    data = data.join(df_cat)
    return data

# Droping the duplicates
def drop_dupli(data):
    data = data.drop_duplicates()
    return data

def conversion(data, column, new_column):
    data[new_column] = data[column] * data.static_usd_rate
    return data




## Prepare data for model training 

# Define target and features

def target(row):
    if row.state == "successful":
        return 1
    elif row.state in ["failed", "suspended"]:
        return 0
    else:
        return np.nan

def create_target(data, column):
    data[column] = data.apply(lambda row:target(row), axis=1)
    data.dropna(axis=0, inplace=True)
    return data


def spliting(data, target, features, rate=0.3):
    y = data[target]
    X = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = rate, random_state = 42, stratify = y)
    return X_train, X_test, y_train, y_test 


def make_model(X_train, X_test, y_train, y_test, clf):
    return clf.fit(X_train, y_train) 
    
def transform(chosen_transformer, X_train, X_test):
    # Prepare list of numerical and categorical columns
    num_cols = make_column_selector(dtype_include=np.number)
    cat_cols = make_column_selector(dtype_include="category")

    transformer =  chosen_transformer([        
        ("scale", StandardScaler(), num_cols),
        ("encode", OneHotEncoder(drop="first"), cat_cols),
    ])

    X_train_trans = transformer.fit_transform(X_train)
    X_test_trans = transformer.transform(X_test)

    return X_train_trans, X_test_trans



### Call funktion etc
#filename = 'data/Kickstarter.csv'
filename = sys.argv[1]
print(filename)
df =  load_data(filename)
print(df.info())
print(df.head(2))

out = ['urls','source_url','currency_symbol', 'currency_trailing_code', 'friends','is_backing','is_starred',
'permissions', 'photo', 'name', 'blurb', 'profile', 'creator', "location", 'slug', "usd_type"]
drop_columns(df, out)

dates = ['created_at', 'launched_at', 'state_changed_at', 'deadline']
df = calc_datetime(df, dates)

categorical = ['country', 'currency','current_currency', 'spotlight',
'staff_pick','state', 'disable_communication', 'is_starrable']
obj_to_categorical(df, categorical)

df = ext_dict(df, 'category')
df = obj_to_categorical(df, 'parent_category_name')

drop_dupli(df)

# additional features:
df['duration'] =  (df.deadline - df.launched_at).dt.days.astype('int')
df['launch_wip'] =  (df.launched_at - df.created_at ).dt.days.astype('int')
df['prep_time'] =  (df.launched_at - df.created_at ).dt.days.astype('int')

conversion(df, 'goal', 'usd_goal')

df["log_usd_goal"] = np.log10(df.usd_goal)
df["pledged_average"] = df.usd_pledged / df.backers_count
df["log_pledged_average"] = np.log10(df.pledged_average)

# Define train and test data
df['success'] = df.state == 'successful'
create_target(df, 'successful')

features = ['usd_goal', 'disable_communication', 'country', 'duration', 
'prep_time', 'parent_category_name', 'category_name']


X_train, X_test, y_train, y_test = spliting(df, 'successful', features, rate=0.3)

X_train, X_test = transform(ColumnTransformer, X_train, X_test)


# ## Simple Logistic Regression with Standard Scaling
logreg_ss = LogisticRegression(max_iter=400)
model = make_model(X_train, X_test, y_train, y_test, logreg_ss)

## saves
filename = 'model.csv' # alternativ save as .sav, works as well
pickle.dump(model, open(filename, 'wb'))

filename = 'train.csv' 
pickle.dump([X_train, y_train], open(filename, 'wb'))

filename = 'test.csv' 
pickle.dump([X_test, y_test], open(filename, 'wb'))

# python file.py 'data/Kickstarter.csv'