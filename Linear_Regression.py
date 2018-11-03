import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Simple_Linear_Regression/Salary_Data.csv')
X = data.iloc[:, 0:1].values
y = data.iloc[:, 1:2].values



'''For splitting the data into training and test set'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.66, random_state = 0)



'''Feature Scaling is necessary for scaling the data.  Normalization(values between 0 and 1)- Use MinMaxScaler
and Standardization(mean = 0 and std. dev=1)'''
'''from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)'''

'''Linear Regression'''
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_predict = linear_regressor.predict(X_test)

'''Regression line for training data and pred training data
Scatterplot for test data. Calculate the error between y_test(red) 
and the y_predict(found above  and corresponding point on graph)'''   
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, linear_regressor.predict(X_train), color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()