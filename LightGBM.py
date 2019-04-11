from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score

from lightgbm import LGBMClassifier
classifier=LGBMClassifier(n_estimators=3000,random_state=1994)
classifier.fit(df.drop(X_train,y_train,eval_metric=['auc'],verbose=100,categorical_feature=[0,1])
y_pred = classifier.predict(X_test)
