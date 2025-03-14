# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.  Set variables for assigning dataset values.
3.  Import linear regression from sklearn.
4.  Assign the points for representing in the graph.
5.  Predict the regression for marks by using the representation of the graph.
6.  Compare the graphs and hence we obtained the linear regression for the given datas.
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Priyanka K
RegisterNumber: 212223230162
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/content/student_scores.csv')
print(df.head())
print(df.tail())

X = df.iloc[:,:-1].values
print(X)
Y = df.iloc[:,-1].values
print(Y)

print(X.shape)
print(Y.shape)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)

MSE = mean_squared_error(Y_test,Y_pred)
print('MSE = ',MSE)
MAE = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',MAE)
RMSE = np.sqrt(MSE)
print('RMSE = ',RMSE)

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred,color='green')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/309346dc-9bd3-41a1-96ed-839305d70fb8)
![image](https://github.com/user-attachments/assets/8da9797b-7196-4018-80c6-fca5acaf1b4f)
![image](https://github.com/user-attachments/assets/9fe9e4a5-a861-4eea-a95a-85d00613f0c2)
![image](https://github.com/user-attachments/assets/5c3d7044-ae1d-4149-a676-626c835b5af5)
![image](https://github.com/user-attachments/assets/5395986d-3670-4af9-a873-4d90c65707cb)
![image](https://github.com/user-attachments/assets/7cde7ac9-6431-4831-b8ac-c90531863258)
![image](https://github.com/user-attachments/assets/0a4b57d1-a337-48ca-8e9b-a1b51e7dfe63)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
