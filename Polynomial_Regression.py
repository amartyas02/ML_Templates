import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Polynomial_Regression/Polynomial_Regression/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
'''Here for X, we could have written [:,1] but then it is a vector not a matrix. To convert it into a matrix, it is [:,1:2]'''
y = data.iloc[:, 2:3].values

'''Here we are not splitting the dataset as we have only 10 observations. Also in Linear Regression, PreScaling is already done.'''

'''Linear Regression Model'''
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y)

'''Polynomial Regression Model'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly,y)

'''Visualizing the linear regression results.'''
plt.scatter(X,y, color ='red')
plt.plot(X,linear_reg.predict(X), color='blue')
plt.show()

'''Visualizing the multiple regression  results.'''
plt.scatter(X,y, color ='red')
plt.plot(X,linear_reg2.predict(poly_reg.fit_transform(X)), color='blue')
'''Above we could have used X_poly insteadof poly_reg.fit_transform(X)'''
plt.show()

'''Predicting the results'''
linear_reg.predict(6.5)
linear_reg2.predict(poly_reg.fit_transform(6.5))