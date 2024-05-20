import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
dataset=pd.read_csv('/content/Social_Network_Ads (3).csv')
print(dataset.columns)
x=dataset[['Age','EstimatedSalary']]
y=dataset['Purchased']
a_train,a_test,b_train,b_test=train_test_split(x,y,test_size=0.25)
#feature scaling
sc=StandardScaler()
a_train=sc.fit_transform(a_train)
a_test=sc.transform(a_test)
classifier=LogisticRegression()
classifier.fit(a_train,b_train)
b_pred=classifier.predict(a_test)
print("confusion matrix")
print(confusion_matrix(b_test,b_pred))
print("Accuracy score",accuracy_score(b_test,b_pred))
sns.regplot(x=a_test[:,:-1],y=b_test,logistic=True)
