# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages.
2.Read the given csv file and display the few contents of the data.
3.Assign the features for x and y respectively.
4.Split the x and y sets into train and test sets.
5.Convert the Alphabetical data to numeric using CountVectorizer.
6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7.Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SRI MATHI S
RegisterNumber:  212224230272
*/
```
~~~
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
~~~

## Output:
## data.head()
![279768985-f25a1488-2cf4-401c-b9a1-c718528f7009](https://github.com/user-attachments/assets/7cc96077-b71a-4327-b06d-2698dfe7742c)
## data.info()
![279769009-701258ca-4918-4d2f-b242-e292c007eb6b](https://github.com/user-attachments/assets/88ca4137-7963-49ab-985b-f06ac5f1d49e)
## data.isnull().sum()

![279769021-f630a7bd-8b5b-4a6b-9339-efc41656d1a6](https://github.com/user-attachments/assets/93ad35d1-adde-466c-9b00-48e01f01d843)
## y_prediction value
![279769054-9f5f562a-6191-47cd-a533-57677ea21f50](https://github.com/user-attachments/assets/0a6024fd-8803-4795-bb1c-2f571a6c15cb)
## accuracy value
![279769089-22fc6d5f-3a59-4a8f-bf6e-5cc76de836a0](https://github.com/user-attachments/assets/803c1936-001d-4d8a-ac87-0d6a1bd3cc15)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
