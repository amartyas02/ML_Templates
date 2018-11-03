import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Decision_Tree_Regression/Decision_Tree_Regression/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
'''Here for X, we could have written [:,1] but then it is a vector not a matrix. To convert it into a matrix, it is [:,1:2]'''
y = data.iloc[:, 2:3].values

from sklearn.ensemble import RandomForestRegressor
'''We use gaussian kernel,which is by default'''
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)

'''Visualization'''

'''High Resolution'''
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
'''High Resolution'''

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color ='blue')
plt.show()

'''Decision tree divides the dataset with the help of boundaries. Here in case of only 1 feature,
it takes the values as intervals and gives the same value(avg. of the values to all in the same interval.
Thus it should be a straight line as in this case.'''
'''Decision Tree is a non-linear, non-continuous model as same value in an interval and then discontinuity.'''

'''Random Forest Regression takes the avg.of results of several Decision Trees and more the no. of trees better more
the avg. of the d/f predictions made by the trees is converging to the same avg.'''

'''By having several decision trees, we get the more number of steps in the stairs as compared to 1 decision tree.Thus
we have a lot more splits of the whole range of levels.'''
