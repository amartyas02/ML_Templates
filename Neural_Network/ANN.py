import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('/home/amartya/Desktop/Artificial_Neural_Networks/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

'''We do not want the 2 encoded to be of higher priority than 1 encoded. So we use one-hot encoding.'''
'''Here we are using onehotencoder only for country column as it has 3 countries and after we remove 1 dummy varible(dummy variable trap)
2 will remain. However in case of Gender 2-1 = 1. So no need for onehotencoding.'''
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
'''Avoiding Dummy variable trap'''
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 0)

'''Feature Scaling'''
from sklearn.preprocessing import StandardScaler
standard_scaler =  StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

'''We have to run till above always.'''
# Method1
#initialising the ANN
classifier = Sequential()
#Adding the input layer with droput
'''Generally we want the hidden layers to be of x nodes, where x = (no_input + no_output)/2'''
'''For hidden layers, we generally choose relu activation function'''
classifier.add(Dense(output_dim = 6, activation='relu', kernel_initializer='uniform', input_dim = 11))
classifier.add(Dropout(rate = 0.1))
#Adding first hidden layer
'''No need of writing input_dim, although is equal to 6.'''
classifier.add(Dense(output_dim = 6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate = 0.1))

#Adding output layer
'''For 1 output class, we use sigmoid function. For more than 1 output category, use softmax activation function.'''
classifier.add(Dense(output_dim = 1, activation='sigmoid', kernel_initializer='uniform'))

#Compiling the ANN
'''We use adam function for stochastic gradient descent. For  binary, we use loss function as binary_crossentropy.'''
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to training set.
classifier.fit(X_train, y_train, batch_size=10, epochs = 50)

#Predicting the Test set 
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Making the confusion metrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN
#Method2
'''KerasClassifier expects one of it's argument to be a function.'''
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, activation='relu', kernel_initializer='uniform', input_dim = 11))
    classifier.add(Dense(output_dim = 6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(output_dim = 1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size =10, nb_epoch = 50)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

'''If we get high variance, that means overfitting has taken place.'''
'''We use Dropout Regularization to overcome overfitting. If overfitting, start with droput of 0.1 and then increase.'''

#Hyperparameter Tuning
#Method3 
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



