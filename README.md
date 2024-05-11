# EXPERIMENT 3:Implementation-of-Linear-Regression-Using-Gradient-Descent

## Aim:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Step 1:
Import the necessary libraries.

Step 2:
Define linear regression function.

Step 3:
Begin by reading the dataset using the read_csv function.

Step 4:
Apply the linear_regression function to the standardized input features X1_Scaled and target variable Y1_Scaled to obtain the optimal parameters theta.

Step 5:
Prepare new data and make predictions using the trained model.

Step 6:
Print the predicted value obtained from the regression analysis.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: AJAY ASWIN M
RegisterNumber: 212222240005

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
# Add a column of ones to X for the intercept term
  X = np.c_[np.ones(len(X1)), X1]
# Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
# Perform gradient descent
  for _ in range(num_iters):

# Calculate predictions
    predictions = (X).dot (theta).reshape(-1, 1)
# Calculate errors
    errors = (predictions - y).reshape(-1,1)
# Update theta using gradient descent
    theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
  return theta
data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())
# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
# Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)
# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:

### Dataset:
![ml exp3-1](https://github.com/Gopika-9266/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122762773/089c011c-110f-4838-b0f2-cc80ac7705cc)

### Predicted value:
![ml exp3-2](https://github.com/Gopika-9266/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122762773/4ff0e362-fb8f-42f1-88ac-bff290cdd8af)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
