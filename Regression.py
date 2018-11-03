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


'''DECISION TREE'''

'''Decision tree divides the dataset with the help of boundaries. Here in case of only 1 feature,
it takes the values as intervals and gives the same value(avg. of the values to all in the same interval.
Thus it should be a straight line as in this case.'''
'''Decision Tree is a non-linear, non-continuous model as same value in an interval and then discontinuity.'''

from sklearn.tree import DecisionTreeRegressor
'''We use gaussian kernel,which is by default'''
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


'''RANDOM FOREST'''

'''Decision tree divides the dataset with the help of boundaries. Here in case of only 1 feature,
it takes the values as intervals and gives the same value(avg. of the values to all in the same interval.
Thus it should be a straight line as in this case.'''
'''Decision Tree is a non-linear, non-continuous model as same value in an interval and then discontinuity.'''

'''Random Forest Regression takes the avg.of results of several Decision Trees and more the no. of trees better more
the avg. of the d/f predictions made by the trees is converging to the same avg.'''

'''By having several decision trees, we get the more number of steps in the stairs as compared to 1 decision tree.Thus
we have a lot more splits of the whole range of levels.'''


from sklearn.ensemble import RandomForestRegressor
'''We use gaussian kernel,which is by default'''
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)


'''SVR'''


'''Feature Scaling is not by default.'''
from sklearn.preprocessing import StandardScaler
standard_scaler_X = StandardScaler()
standard_scaler_y = StandardScaler()

X = standard_scaler_X.fit_transform(X)
y = standard_scaler_y.fit_transform(y)


from sklearn.svm import SVR
'''We use gaussian kernel,which is by default'''
regressor = SVR(kernel='rbf')
regressor.fit(X,y)


'''Suppose we wanted to find the value of y for x = 6.5. First of all since feature scaling is there,we need to transform 6.5
Then we  find that the input of transform should be an array. Hence through np.array, we convert 6.5 to array. Now we get the 
PreScaled result. Thus through inverse_transform, we get the original value.'''
pred_y = standard_scaler_y.inverse_transform(regressor.predict(standard_scaler_X.transform(np.array([[6.5]]))))


#After performing any one, visualize the rsults.

'''Visualization'''

'''High Resolution'''
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
'''High Resolution'''

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color ='blue')
plt.show()









