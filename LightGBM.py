import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)
params = {}

params['learning_rate'] = 0.003
#‘boosting type’ is gbdt, we are implementing gradient boosting(you can try random forest)
params['boosting_type'] = 'gbdt'
#Used ‘binary’ as objective(remember this is classification problem)
params['objective'] = 'binary'
#Used ‘binary_logloss’ as metric(same reason, binary classification problem)
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
#‘num_leaves’=10 (as it is small data)
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)

y_pred=clf.predict(X_test)
#convert into binary values
for i in range(0,99):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)