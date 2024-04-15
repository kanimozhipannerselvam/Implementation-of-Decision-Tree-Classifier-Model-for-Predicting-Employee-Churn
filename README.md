
# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KANIMOZHI
RegisterNumber:  212222230060
*/
```
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
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
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:

## HEAD() AND INFO():
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118610231/f8005b20-85c6-444d-86ef-05988de2914d)

## NULL & COUNT:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118610231/6227bcd1-239e-4610-b769-27294162049a)

![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118610231/e26eeddb-0b03-42c5-955b-860c919822b0)

## ACCURACY SCORE:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118610231/e67d3205-ec86-4b99-956b-00b9fb923498)

## DECISION TREE CLASSIFIER MODEL:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118610231/7c93ff98-b7b4-455b-aa42-d1c38c6391f2)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
