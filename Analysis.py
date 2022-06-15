#!/usr/bin/env python
# coding: utf-8

# # Fish Weight Analysis and Prediction

# ## 1. Importing Required Libraries

# In[1]:


import pandas as pd # For file operations
import matplotlib.pyplot as plt # For visualization
import numpy as np # For numpy model evaluation
from sklearn.preprocessing import OrdinalEncoder # For handling categorical variables
import statsmodels.api as sm # For model summary

import re # For data wrangling

from sklearn.model_selection import train_test_split # For creating training and testing datasets
from sklearn.linear_model import LinearRegression # For linear regression model
from sklearn.ensemble import RandomForestRegressor # For random forest model
from sklearn import metrics # For metric evaulation

import warnings # To disable warnings
warnings.filterwarnings("ignore")


# ## 2. Data Wrangling

# In[2]:


# Reading dataset

filepath = "C:/Users/dhrru/Downloads/Air Quality/fish_market.csv"
depvar = "Weight"

def perform_regression(filepath, depvar):
    if ".csv" in filepath:
        data = pd.read_csv(filepath, index_col=False, encoding='unicode_escape')
    elif ".xls" in filepath:
        data = pd.read_excel(filepath, index_col=False)
    elif ".tsv" in filepath:
        data = pd.read_table(filepath, index_col=False)
    elif ".json" in filepath:
        data = pd.read_json(filepath)
    else:
        data = pd.read_csv(filepath, index_col=False, sep=" ")
    
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Displaying the dataframe
    print(data)

    # Displaying first 5 rows
    print("\n\n", data.head(5))

    # Displaying last 5 rows
    print("\n\n", data.tail(5))

    # Displaying dataset columns
    print("\n\n", data.columns)

    # Displaying data shape (rows x columns)
    print("\n\n", data.shape)

    # Data description
    print("\n\n", data.describe())

    # Datatype information 
    print("\n\n", data.info())

    print("\n\nData Shape Before", data.shape)
    data.drop_duplicates(keep=False, inplace=True)
    print("Data Shape After", data.shape)

    # Looking for NA/Null values
    print("\n\n", data.isna().sum())

    # Dropping and filling nan values
    data.dropna(how='all')
    data = data.fillna(0)

    # Estimating coorelation
    correlation = data.corr().abs()
    print("\n\n", correlation)

    # Finding highly coorelated variables
    highly_correlated_variables = np.where(correlation>0.8)
    highly_correlated_variables = [(correlation.columns[x],correlation.columns[y]) for x, y in zip(*highly_correlated_variables) if x!=y and x<y]
    highly_correlated_variables = [re.sub(r'\([^)]*\)', '', x) for x in highly_correlated_variables[0]]
    print("\n\n", highly_correlated_variables)

    # Finding categorical columns
    categorical = data.select_dtypes(exclude=["number","bool"])
    print("\n\nNumerical Columns: ", len(data.columns)-len(categorical.columns))
    numerical = len(data.columns)-len(categorical.columns)
    print("Categorical Columns: ", len(data.columns)-numerical)

    # Displaying categorical columns
    print("\n\n", data[list(categorical.columns)])

    # Encoding categorical columns
    encoder = OrdinalEncoder()
    data[list(categorical.columns)] = encoder.fit_transform(data[list(categorical.columns)])
    print("\n\n", data)

    # Comparing linear regression with random forest regression
    train = data.loc[:,data.columns !=depvar]
    test = data.loc[:,data.columns ==depvar]
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=test_size, random_state=0)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    linear_regression_prediction = linear_regression.predict(X_test)
    print("Linear Regression:")
    print("\nCoefficients: ", linear_regression.coef_)
    print("Variance score: {}".format(linear_regression.score(X_test, y_test)))
    print("\nMean Absolute Error: ", metrics.mean_absolute_error(y_test,linear_regression_prediction) )
    print("Measn Square Error: ", metrics.mean_squared_error(y_test,linear_regression_prediction))
    print("Root Mean Square Error: ", np.sqrt(metrics.mean_squared_error(y_test, linear_regression_prediction)))
    lr_model_coefficient=pd.DataFrame(linear_regression.coef_[0],train.columns)
    lr_model_coefficient.columns = ['Coefficient']
    print("\n", lr_model_coefficient)
    plt.style.use('fivethirtyeight')
    plt.scatter(y_test,linear_regression_prediction, color = 'aqua')
    plt.title('Actual vs Predicted Data')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, y_train)
    random_forest_prediction = random_forest.predict(X_test)
    print("\n\nRandom Forest:")
    print("Variance score: {}".format(random_forest.score(X_test, y_test)))
    print("\nMean Absolute Error: ", metrics.mean_absolute_error(y_test,random_forest_prediction) )
    print("Measn Square Error: ", metrics.mean_squared_error(y_test,random_forest_prediction))
    print("Root Mean Square Error: ", np.sqrt(metrics.mean_squared_error(y_test, random_forest_prediction)))
    plt.style.use('fivethirtyeight')
    plt.scatter(y_test,random_forest_prediction, color = 'red')
    plt.legend(labels = ('Linear Regression','Random Forest'),loc='upper left')
    plt.show()
    
    model = sm.OLS(y_train, X_train)
    ols = model.fit()
    print("\nModel Summary:")
    print(ols.summary())

perform_regression(filepath, depvar)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




