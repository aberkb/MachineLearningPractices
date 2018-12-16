#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:47:13 2018

@author: aberkb
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
# Reading dataset using pandas
titanic_dataset = pd.read_csv('titanic_dataset.csv')
# Filling missing age data according Passenger class
def avarage_age(columns):
    Age = columns[0]
    Pclass = columns[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


titanic_dataset['Age'] = titanic_dataset[['Age', 'Pclass']].apply(avarage_age, axis=1)

# Cabin column %90 missing so i am just gonna drop this feature
titanic_dataset.drop('Cabin', axis=1, inplace=True)
# Embark and sex columns are categorical values so i am converting them using get_dummies
sex = pd.get_dummies(titanic_dataset['Sex'], drop_first=True)
embark = pd.get_dummies(titanic_dataset['Embarked'], drop_first=True)
titanic_dataset.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
titanic_dataset = pd.concat([titanic_dataset, sex, embark], axis=1)

# setting features and labels(whether they survived or not)
X = titanic_dataset.drop('Survived', axis=1)
y = titanic_dataset['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30,
                                                    random_state=55)


print("LOGISTIC REGRESSION REPORT AND CONF. MATRIX")
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
preds = logistic_reg.predict(X_test)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test,preds))

print("SUPPORT VECTOR MACHINE REPORT AND CONF. MATRIX")
svclassifier = SVC()
svclassifier.fit(X_train,y_train)
svm_preds = svclassifier.predict(X_test)
print(classification_report(y_test,svm_preds))
print(confusion_matrix(y_test,svm_preds))

print("RANDOM FOREST REPORT AND CONF. MATRIX")
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))

print("K-NEAREST NEIGHBORS REPORT AND CONF. MATRIX")
knn = KNeighborsClassifier(n_neighbors=90)
knn.fit(X_train,y_train)
knn_preds = knn.predict(X_test)
print(classification_report(y_test,knn_preds))
print(confusion_matrix(y_test,knn_preds))