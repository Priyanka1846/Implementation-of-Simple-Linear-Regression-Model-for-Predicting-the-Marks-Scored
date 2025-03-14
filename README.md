# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Priyanka K
RegisterNumber: 212223230162
*/
```
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
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
