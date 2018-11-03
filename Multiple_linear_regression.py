import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Multiple_Linear_Regression/50_Startups.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
'''Avoiding the dummy variable trap.In case of multiple linear regression, avoid 1 column of 1hot encoder.
This is because thereare 2 dependent variables. So we take 1 less.'''
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)
'''Same as in linear regression considering weights to be equal.'''
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)

#If we consider weights also, then we have to calculate them.
import statsmodels.formula.api as sm
'''To use statsmodels,  we  need to add an extra column of x0 =1'''
X = np.append(arr = np.ones((50,1)).astype(int), values = X,  axis =1)

'''Assuming p-value threshold to be 0.5, we remove all variables with p-value >0.5.
However we can first run and then also decide the threshold.'''
'''OLS is ordinary least squares.'''
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
'''We observe that the p-value of x2 is highest. Thus we remove it.'''

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
'''Now remove x1 from the remaining indices i.e [1]'''

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
'''Now remove x2'''

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
'''Thus we can say that the major factors are constant and R&D spending.'''
'''In such cases, check for  R-squared error. The  more R-squared error wins.'''




