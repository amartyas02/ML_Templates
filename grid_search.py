import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/K_Nearest_Neighbors/K_Nearest_Neighbors/Social_Network_Ads.csv')

X = data.iloc[:, 2:4].values
y = data.iloc[:, 4:5].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.75, random_state = 0)

'''Feature Scaling'''
from sklearn.preprocessing import StandardScaler
standard_scaler =  StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

'''Kernel_SVM(Max.length classifier)'''
'''It is a gaussian curve that is at the maximum sum of distances from the 2 nearest points(support vectors)
to it. It looks at the extreme cases which are very close to the boundary and uses that to construct it's analysis. 
Same as SVM, just we use gaussian kernel. '''

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train, y_train)

'''Predicting the test set result'''
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''Applying k-fold cross validation'''
'''
This is much better than accuracy through confusion matrix as it first divides 
the training set into cv(here 10) parts and then determines the accuracy of each part.
''' 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

'''Applying grid-search to find best model and best parameters.'''
from sklearn.model_selection import GridSearchCV
# For parameters, check the parameters of the classifier(here SVC) and make it a list of dictionaries.
parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.5,0.1,0.01,0.001,0.50]}
              ] 
'''In SVC, C is the regularization parameter. It prevents overfitting. If it is too large, then overfitting can occur.
Kernel specifies whether we use linear, rbf or poly kernel. 
Gamma parameter is (1/no.of features, here 0.5) kernel coefficient.
Choose the parameters around the default and then after the result, change the calues and test again.
'''
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring='accuracy', cv =10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
'''Visualizing the results on the training set.'''

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
plt.title('SVM(Training Set)')
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
plt.title('KNN(Training Set)')
plt.xlabel('Age')
plt.ylabel('Salary') 
plt.legend()   
plt.show()