# Kickstarter

![Title Image](https://github.com/lima-tango/kickstarter/blob/main/images/title.jpg "Thanks to [Med Badr Chemmaoui](https://unsplash.com/photos/ZSPBhokqDMc)")

## Table of Contents

1. [Installation](#installation)
1. [Motivation](#motivaton)
1. [File Descriptions](#files)
1. [Data Description](#data)
1. [Results](#results)
1. [Model](#model)

## Installation <a name="installation"></a>

To execute this notebook, you need python 3.8.5.
All packages can be installed with:
```bash
pip install -r requirements
```

## Motivation <a name="motivation"></a>

In recent years, the range of funding options for projects created by individuals and small companies has expanded considerably. In addition to savings, bank loans, friends & family funding, and other traditional options, crowdfunding has become a popular and readily available alternative.

Kickstarter, founded in 2009, is one particularly well-known and popular crowdfunding platform. It has an all-or-nothing funding model, whereby a project is only funded if it meets its goal amount; otherwise no money is given by backers to a project.
A huge variety of factors contribute to the success or failure of a project — in general, and also on Kickstarter. Some of these are able to be quantified or categorized, which allows for the construction of a model to attempt to predict whether a project will succeed or not. The aim of this project is to construct such a model and also to analyse Kickstarter project data more generally, in order to help potential project creators assess whether or not Kickstarter is a good funding option for them, and what their chances of success are.

## File Descriptions <a name="files"></a>

| File | Description |
| --- | --- |
| EDA.ipynb | Analysis of the data. |
| Model.ipynb | Development of the best model. |
| | A python script that prepares the data, trains a model, and predicts on test data. |
| | |

## Data Description <a name="data"></a>

A detailed description of the data is given [here](https://github.com/lima-tango/kickstarter/blob/main/columns.md)

## The Notebook <a name="The Notebook"></a>

The Jupyter notebook (**Kickstarter_project_group3.ipynb**) contains the data cleaning, feature engineering, a detailed EDA and an overview of the consideration of different models, with different hyperparameters.

Due to github storage limitation the uploaded notebooks contains no output, please run them yourself.


## Python scripts <a name="Python scripts"></a>

There are two python scripts, the first one (**generate_model.py**) for building the best model found in the notebook and the second (**run_model.py**) for runing the prediction.

This repository contains two python scripts.
The first one (**generate_model.py**), takes the dataset with path as argument. It calculates the best model, choosen for accuracy, precision and simplicity in the notebook. Afterwards it saves the model (*model.csv*, contains the model for further calculation) and the test (*test.csv*, contains X_test, y_test) and train (*train.csv*, contains X_train, y_train) files.

The second script (**run_model.py**), takes the saves from the first. It calculates the prediction, the classification_report, the confusion matrix, the latter it also plots. As output it saves the predictions (*prediction.csv*, contains y_train_pred, y_pred) and the classification_report (*class_report.csv*, containing the report with the evaluation metrics)

#### Commands to run the scripts:
```bash
python generate_model.py 'data/Kickstarter.csv'
python run_model.py 'test.csv' 'train.csv' 'model.csv' 
```
## Results <a name="results"></a>

The results are presented in this [presentation](https://github.com/lima-tango/kickstarter/blob/main/kickstarter_presentation.pdf).


