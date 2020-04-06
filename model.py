# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

df = pd.read_csv('dataset.csv')
df['TotalCharges']=df['TotalCharges'].replace(r'\s',np.nan, regex=True)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'])

fill=df.MonthlyCharges * df.tenure
df.TotalCharges.fillna(fill,inplace=True)
df.isnull().sum()

df=pd.get_dummies(df,columns=['Partner','Dependents','PhoneService','MultipleLines','StreamingTV','StreamingMovies','Contract','PaperlessBilling','InternetService'],drop_first=True)

df.drop(['StreamingTV_No internet service','StreamingMovies_No internet service'],axis=1,inplace=True)

df.drop('gender',axis=1,inplace=True)
df.drop('customerID',axis=1,inplace=True)
df.drop(['tenure','MonthlyCharges'],axis=1,inplace=True)
df.drop(['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','PaymentMethod'],axis=1,inplace=True)
df=pd.get_dummies(df,columns=['Churn'],drop_first=True)


X = df.drop('Churn_Yes',axis=1).as_matrix().astype('float')
Y = df['Churn_Yes'].ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(random_state=0)
model.fit(X_train,Y_train)




pickle.dump(model, open('model.pkl','wb'))


print(model.predict([[2, 9, 6]]))
