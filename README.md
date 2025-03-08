# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters – Set initial values for slope m and intercept 𝑏 and choose a learning rate 𝛼
2. Compute Cost Function – Calculate the Mean Squared Error (MSE) to measure model performance.
3. Update Parameters Using Gradient Descent – Compute gradients and update m and b using the learning rate. 
4. Repeat Until Convergence – Iterate until the cost function stabilizes or a maximum number of iterations is reached.

## Program and Output:

Program to implement the linear regression using gradient descent.

Developed by: DHIRAVIYA S

RegisterNumber:  212223040041

```
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)

        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\ML\\DATASET-20250226\\50_Startups.csv")
data.head()
```

![image](https://github.com/user-attachments/assets/13701738-78df-4706-bee2-340b4f256880)

```
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
```

![image](https://github.com/user-attachments/assets/84fe68af-607b-4c4c-ae53-3e428846f9ee)

```
print(X1_Scaled)
```

![image](https://github.com/user-attachments/assets/ab8f9e2a-6c29-42fc-90b0-ae9a87074c21)


```
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
![image](https://github.com/user-attachments/assets/59c78e52-78db-41f0-8e68-46c8c5714fde)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
