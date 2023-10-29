# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data Clean and format your data Split your data into training and testing sets
2. Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms
3. Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions
4. Define your learning rate Determines how quickly weights are updated during gradient descent
5. Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations
6. Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score
7. Tune hyperparameters Experiment with different learning rates and regularization techniques
8. Deploy your model Use trained model to make predictions on new data in a real-world application.

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JAYABHARATHI S
RegisterNumber:  212222100013
*/

import pandas as pd
df=pd.read_csv("/content/Employee.csv")

df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

## Initial data set :
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/c25c2ecc-b41a-4f61-8838-3eb6e95289c4)

## Data info:
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/7c5eff2f-1af8-41a5-97bd-0c0df81bb397)

## Optimization of null values:
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/d6ac42b4-da1e-4ad7-a0bd-eb388b649414)

## Assignment of x and y values:
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/2074591e-0d3e-4edb-9b3b-bcd39b17db4a)

![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/9627743d-71b8-47f0-be87-6518bbd431a8)

## Converting string literals to numerical values using label encoder:
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/443adbb8-503e-458a-b76c-da0a131a6e93)

##  Accuracy:
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/f34498c2-850e-4cce-ba8e-25982b398a55)

## Prediction:
![image](https://github.com/Jayabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120367796/76b996ff-b7dd-4875-a3b6-f316a9fccef5)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
