import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Data_Preprocessing/Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

'''Imputer is used to fill missing values.'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

'''LabelEncoder is used to label categorical values. Converts into a array'''
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:, 0])

'''OneHotEncoder is used to label numerical values after labelling. Can be used only after labelencoder. 
Converts into matrix. Thus has to be converted to array before proceeding further.'''
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
y = labelencoder.fit_transform(y)

'''For splitting the data into training and test set'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)

'''Feature Scaling is necessary for scaling the data.  Normalization(values between 0 and 1)- Use MinMaxScaler
and Standardization(mean = 0 and std. dev=1)'''
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)