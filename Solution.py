import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score



data = pd.read_csv('/home/amartya/Desktop/Yes_Bank_Training.csv')
'''
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
'''
'''
X = pd.DataFrame(X)
    
X.columns = ['serial_number', 'age_in_years', 'job_description', 'marital_status',
       'education_details', 'has_default', 'balance_in_account',
       'housing_status', 'previous_loan', 'phone_type', 'date',
       'month_of_year', 'call_duration', 'campaign_contacts', 'days_passed',
       'previous_contact', 'poutcome_of_campaign']
'''

temp = data.replace({'job_description' : {'admin.': 8,  
 'blue-collar': 2,  
 'entrepreneur': 10, 
 'housemaid': 1,  
 'management': 12,  
 'retired': 3,  
 'self-employed': 11,  
 'services': 7,  
 'student': 9 ,  
 'technician': 6,  
 'unemployed': 5,  
 'unknown': 4}})   

temp = temp.replace({'education_details' : {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 4}})
temp = temp.replace({'marital_status' : {'divorced': 1,  
 'married': 2,  
 'single': 3
 }})
temp = temp.replace({'month_of_year' : {'apr': 1,  
 'aug': 2,  
 'feb': 3,  
 'jan': 4,  
 'jul': 5,  
 'jun': 6, 
 'mar': 7, 
 'may': 8,  
 'nov': 9,  
 'oct': 10,
 'dec':11,
 'sep':12
 }})
temp = temp.replace({'has_default' : {'no': 1,  
 'yes': 2}})
temp = temp.replace({'housing_status' : {'no': 1,  
 'yes': 2}})
temp = temp.replace({'previous_loan' : {'no': 1,  
 'yes': 2}})
temp = temp.replace({'outcome' : {'no': 1,  
 'yes': 2}})



lis = []
for i in range(len(temp)):
    if temp.iloc[i][4]==4 :
        lis.append(i)
        
for n in lis:
    temp = temp.drop(n, axis = 0)
    
#Changing in data to fill the education columns

data_ = data.replace({'job_description' : {'admin.': 8,  
 'blue-collar': 2,  
 'entrepreneur': 10, 
 'housemaid': 1,  
 'management': 12,  
 'retired': 3,  
 'self-employed': 11,  
 'services': 7,  
 'student': 9 ,  
 'technician': 6,  
 'unemployed': 5,  
 'unknown': 4}})   

data_ = data_.replace({'education_details' : {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 4}})
data_ = data_.replace({'marital_status' : {'divorced': 1,  
 'married': 2,  
 'single': 3
 }})
data_ = data_.replace({'month_of_year' : {'apr': 1,  
 'aug': 2,  
 'feb': 3,  
 'jan': 4,  
 'jul': 5,  
 'jun': 6, 
 'mar': 7, 
 'may': 8,  
 'nov': 9,  
 'oct': 10,
 'dec':11
 }})
data_ = data_.replace({'has_default' : {'no': 1,  
 'yes': 2}})
data_ = data_.replace({'housing_status' : {'no': 1,  
 'yes': 2}})
data_ = data_.replace({'previous_loan' : {'no': 1,  
 'yes': 2}})
data_ = data_.replace({'outcome' : {'no': 1,  
 'yes': 2}})    
    
data_ = data_.replace({'poutcome_of_campaign' : {'failure': 1,
                                                'other': 2,
                                                'success': 3,
                                                'unknown': 4}})
for j in lis:
   if data_.iloc[j][4] == 4:
       if data_.iloc[j][2] == 1:
           data_.at[j, 'education_details'] = 1
       elif (data_.iloc[j][2] == 2 or data_.iloc[j][2] == 3 or data_.iloc[j][2] == 4 or data_.iloc[j][2] == 5 or data_.iloc[j][2] == 6 or data_.iloc[j][2] == 7 or data_.iloc[j][2] == 8 or data_.iloc[j][2] == 9):
           data_.at[j, 'education_details'] = 2
       elif (data_.iloc[j][2] == 10 or data_.iloc[j][2] == 11 or data_.iloc[j][2] == 12):
           data_.at[j, 'education_details'] = 3
       else :
           pass
data_ = data_.drop('phone_type', axis =1 )
data_ = data_.drop('serial_number', axis =1 )
data_ = data_.drop('date', axis = 1)
data_ = data_.drop('month_of_year', axis = 1)



##pOUTCOME
         
 '''      
li = []
for i in range(len(pout)):
    if pout.iloc[i][16]==4 :
        li.append(i)
        
for n in li:
    pout = pout.drop(n, axis = 0)
    
c = pout.corr()
'''
'''
{'cellular': 17181, 'telephone': 1703, 'unknown': 12765}
'''
'''
days passed : -1 : 31648 WE can take mean or mode accordingly. Many no.s
'''
'''
 {'failure': 1439, 'other': 538, 'success': 81, 'unknown': 29591}
'''
'''from pandas.plotting import scatter_matrix
scatter_matrix(temp)'''


'''m = pd.crosstab(temp['education_details'],temp['outc'])
for i in m.columns:
    sum = 0
    for j in range(len(m)):
        sum = sum + m.iloc[j][i]
    m[i] = m[i]/sum'''
    
    

x = data_['age_in_years'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_['age_in_years'] = pd.DataFrame(x_scaled)

x = data_['balance_in_account'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_['balance_in_account'] = pd.DataFrame(x_scaled)
'''
x = data_['date'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_['date'] = pd.DataFrame(x_scaled)
'''
x = data_['call_duration'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_['call_duration'] = pd.DataFrame(x_scaled)

x = data_['campaign_contacts'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_['campaign_contacts'] = pd.DataFrame(x_scaled)
      
x = data_['days_passed'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_['days_passed'] = pd.DataFrame(x_scaled)

data_['days_passed'] = data_['days_passed'].replace(0, -1)



#XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

#Naive_Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

##KNN
#from sklearn.preprocessing import StandardScaler
#standard_scaler =  StandardScaler()
#X_train = standard_scaler.fit_transform(X_train)
#X_test = standard_scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

#KernelSVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

#RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

'''Final'''
X = data_.iloc[:, :-1].values
y = data_.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.decomposition import PCA
pca = PCA(n_components = 11)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

''''''
#TEST DATA

data_test = pd.read_csv('/home/amartya/Desktop/Yes_Bank_Test.csv')


temp = data_test.replace({'job_description' : {'admin.': 8,  
 'blue-collar': 2,  
 'entrepreneur': 10, 
 'housemaid': 1,  
 'management': 12,  
 'retired': 3,  
 'self-employed': 11,  
 'services': 7,  
 'student': 9 ,  
 'technician': 6,  
 'unemployed': 5,  
 'unknown': 4}})   

temp = temp.replace({'education_details' : {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 4}})
temp = temp.replace({'marital_status' : {'divorced': 1,  
 'married': 2,  
 'single': 3
 }})
temp = temp.replace({'month_of_year' : {'apr': 1,  
 'aug': 2,  
 'feb': 3,  
 'jan': 4,  
 'jul': 5,  
 'jun': 6, 
 'mar': 7, 
 'may': 8,  
 'nov': 9,  
 'oct': 10,
 'dec':11
 }})
temp = temp.replace({'has_default' : {'no': 1,  
 'yes': 2}})
temp = temp.replace({'housing_status' : {'no': 1,  
 'yes': 2}})
temp = temp.replace({'previous_loan' : {'no': 1,  
 'yes': 2}})
temp = temp.replace({'outcome' : {'no': 1,  
 'yes': 2}})



li = []
for i in range(len(temp)):
    if temp.iloc[i][4]==4 :
        li.append(i)
        
for n in li:
    temp = temp.drop(n, axis = 0)
    
#Changing in data to fill the education columns

data_test_ = data_test.replace({'job_description' : {'admin.': 8,  
 'blue-collar': 2,  
 'entrepreneur': 10, 
 'housemaid': 1,  
 'management': 12,  
 'retired': 3,  
 'self-employed': 11,  
 'services': 7,  
 'student': 9 ,  
 'technician': 6,  
 'unemployed': 5,  
 'unknown': 4}})   

data_test_ = data_test_.replace({'education_details' : {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 4}})
data_test_ = data_test_.replace({'marital_status' : {'divorced': 1,  
 'married': 2,  
 'single': 3
 }})
data_test_ = data_test_.replace({'month_of_year' : {'apr': 1,  
 'aug': 2,  
 'feb': 3,  
 'jan': 4,  
 'jul': 5,  
 'jun': 6, 
 'mar': 7, 
 'may': 8,  
 'nov': 9,  
 'oct': 10,
 'dec':11,
 'sep':12
 }})
data_test_ = data_test_.replace({'has_default' : {'no': 1,  
 'yes': 2}})
data_test_= data_test_.replace({'housing_status' : {'no': 1,  
 'yes': 2}})
data_test_ = data_test_.replace({'previous_loan' : {'no': 1,  
 'yes': 2}})
data_test_= data_test_.replace({'outcome' : {'no': 1,  
 'yes': 2}})    
    
data_test_= data_test_.replace({'poutcome_of_campaign' : {'failure': 1,
                                                'other': 2,
                                                'success': 3,
                                                'unknown': 4}})
for j in li:
   if data_test_.iloc[j][4] == 4:
       if data_test_.iloc[j][2] == 1:
           data_test_.at[j, 'education_details'] = 1
       elif (data_test_.iloc[j][2] == 2 or data_test_.iloc[j][2] == 3 or data_test_.iloc[j][2] == 4 or data_test_.iloc[j][2] == 5 or data_test_.iloc[j][2] == 6 or data_test_.iloc[j][2] == 7 or data_test_.iloc[j][2] == 8 or data_test_.iloc[j][2] == 9):
           data_test_.at[j, 'education_details'] = 2
       elif (data_test_.iloc[j][2] == 10 or data_test_.iloc[j][2] == 11 or data_test_.iloc[j][2] == 12):
           data_test_.at[j, 'education_details'] = 3
       else :
           pass
data_test_ = data_test_.drop('phone_type', axis =1 )
data_test_ = data_test_.drop('serial_number', axis =1) 
data_test_ = data_test_.drop('date', axis = 1)
data_test_ = data_test_.drop('month_of_year', axis = 1)

x = data_test_['age_in_years'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_test_['age_in_years'] = pd.DataFrame(x_scaled)

x = data_test_['balance_in_account'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_test_['balance_in_account'] = pd.DataFrame(x_scaled)
'''
x = data_test_['date'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_test_['date'] = pd.DataFrame(x_scaled)
'''
x = data_test_['call_duration'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_test_['call_duration'] = pd.DataFrame(x_scaled)

x = data_test_['campaign_contacts'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_test_['campaign_contacts'] = pd.DataFrame(x_scaled)
      
x = data_test_['days_passed'].values.astype(float) #returns a numpy array
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler(feature_range=(0,10))
x_scaled = min_max_scaler.fit_transform(x)
data_test_['days_passed'] = pd.DataFrame(x_scaled)

data_test_['days_passed'] = data_test_['days_passed'].replace(0, -1)


#Predicting test results
X = data_.iloc[:, :-1].values
y = data_.iloc[:, -1].values

test_X = data_test_.values

from sklearn.decomposition import PCA
pca = PCA(n_components = 11)
X = pca.fit_transform(X)
test_X = pca.transform(test_X)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)
test_Y = classifier.predict(test_X)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)
test_Y = classifier.predict(test_X)

test_df = pd.DataFrame(test_Y)
test_df = test_df.replace({1: 'no',   2: 'yes'})  

test_df.to_csv('resx.csv', sep=',')



