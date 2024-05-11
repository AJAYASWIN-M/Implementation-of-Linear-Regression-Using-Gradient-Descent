# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## 1.Import Necessary Libraries:
Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

## 2. Define the Linear Regression Function:
Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

## 3. Load and Preprocess the Data:
Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.
## 4. Perform Linear Regression:
Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

## 5.Make Predictions on New Data:
Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

## 6.Print the Predicted Value:
Display the predicted value for the target variable based on the linear regression model applied to the new data.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: AJAY ASWIN M
RegisterNumber:  21222224005
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1)*X.T.dot(errors))
    return theta
data=pd.read_csv('50_startups.csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```

## Output:
![image](https://github.com/saiganesh2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742342/280e29fc-340b-4376-aa79-347c2a86447e)

![image](https://github.com/saiganesh2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742342/2ba3af26-39f6-4a18-a7e7-9d749a13ed20)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
