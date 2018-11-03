import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/SVR/SVR/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
'''Here for X, we could have written [:,1] but then it is a vector not a matrix. To convert it into a matrix, it is [:,1:2]'''
y = data.iloc[:, 2:3].values

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

'''Visualization'''
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color ='blue')
plt.show()

'''Suppose we wanted to find the value of y for x = 6.5. First of all since feature scaling is there,we need to transform 6.5
Then we  find that the input of transform should be an array. Hence through np.array, we convert 6.5 to array. Now we get the 
PreScaled result. Thus through inverse_transform, we get the original value.'''
pred_y = standard_scaler_y.inverse_transform(regressor.predict(standard_scaler_X.transform(np.array([[6.5]]))))