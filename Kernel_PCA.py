import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Logistic_Regression/Logistic_Regression/Social_Network_Ads.csv')

'''Logistic Regression is used when the dependent variable(target) is categorical.'''

X = data.iloc[:, 2:4].values
y = data.iloc[:, 4:5].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.75, random_state = 0)

'''Feature Scaling'''
from sklearn.preprocessing import StandardScaler
standard_scaler =  StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)


'''(Non-Linearly  separable data -> Linearly Separable Data)
Kernel PCA uses kernel trick to map the data to a higher dimension and then apply
PCA to  extract new components that will be the new directions that explains the variance
more. Thus we get new dimensions in  which data will be linearly separable even
by linear classifier.
'''

from sklearn.decomposition import KernelPCA  
kpca = KernelPCA(n_components = 2, kernel='rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

'''Predicting the test set result'''
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''Visualizing the results on the training set.'''

'''This code marks the decision boundary for classification.Decision boundary is straight line bcz logistic regression 
classifier is linear.'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red','green'))(i), label = j)
plt.title('Logistic Regression(Training Set)')
plt.xlabel('Age')
plt.ylabel('Salary') 
plt.legend()   
plt.show()

'''Visualizing the test set results.'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red','green'))(i), label = j)
plt.title('Logistic Regression(Test Set)')
plt.xlabel('Age')
plt.ylabel('Salary') 
plt.legend()   
plt.show()



