# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
import sklearn
import scikitplot as skplt
import matplotlib.pyplot as plt
import sklearn

# Importing the dataset
dataset = pd.read_csv('taxi_dataset_2.csv')
dataset=dataset.dropna(how="any")
print(dataset)
print(dataset.info())

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

print("")
print("Adaboost Algorithm")
model = AdaBoostClassifier(n_estimators=90, random_state=0)
model.fit(X_train, y_train)

# load the model from disk
import pickle
adPickle = open('training_pickle', 'wb')
pickle.dump(model, adPickle)
adPickle.close()


y_pred = model.predict(X_test)
y_pred = y_pred.round()
#confussion Matrix
cmd = confusion_matrix(y_test, y_pred)
print("Confussion Matrix for AdaBoost")
print(cmd)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cmd, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cmd.shape[0]):
    for j in range(cmd.shape[1]):
        ax.text(x=j, y=i,s=cmd[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of Adaboost', fontsize=18)
plt.show()
adb = accuracy_score(y_test, y_pred)

fpr1, tpr1, _ = sklearn.metrics.roc_curve(y_test, y_pred)
auc_score1 = sklearn.metrics.auc(fpr1, tpr1)


print("accuracy score of adaboost algorithm accuracy is ")
print(adb)
print("")


testy = y_test
yhat_classes = y_pred
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)



#XGBoost training
print("XGBoost training")
from xgboost import XGBClassifier


# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 2,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }
            
            
            
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)



# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
print("ypred of XGBoost Algorithm ")
print(y_pred)


#confussion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confussion Matrix for XGBoost Algorithm")
print(cm)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of XGBoost', fontsize=18)
plt.show()
fpr2, tpr2, _ = sklearn.metrics.roc_curve(y_test, y_pred)
auc_score2 = sklearn.metrics.auc(fpr2, tpr2)

xgb = accuracy_score(y_test, y_pred)

print("XGBoost accuracy is ")
print(xgb)
print("")


testy = y_test
yhat_classes = y_pred
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)


plt.figure(figsize=(7, 6))
plt.plot(fpr1, tpr1, color='blue',
label='ROC (ADB AUC = %0.4f)' % auc_score1)
plt.plot(fpr2, tpr2, color='red',
label='ROC (XGB AUC = %0.4f)' % auc_score2)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


data = {'ADABOOST':adb, 'XGBOOST':xgb}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("ML Algorithms")
plt.ylabel("Accuracy")
plt.title("Performance")
plt.show()
print("")
flg=0;

