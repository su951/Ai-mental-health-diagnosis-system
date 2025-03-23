!pip install scikit-learn streamlit pandas

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/dataset.csv')
df = pd.DataFrame(data)
print(df.head())

df['feeling.nervous']=df['feeling.nervous'].astype('category').cat.codes
df['panic']=df['panic'].astype('category').cat.codes
df['breathing.rapidly']=df['breathing.rapidly'].astype('category').cat.codes
df['sweating']=df['sweating'].astype('category').cat.codes
df['anger']=df['anger'].astype('category').cat.codes
df['over.react']=df['over.react'].astype('category').cat.codes
df['trouble.in.concentration']=df['trouble.in.concentration'].astype('category').cat.codes
df['having.trouble.in.sleeping']=df['having.trouble.in.sleeping'].astype('category').cat.codes
df['having.trouble.with.work']=df['having.trouble.with.work'].astype('category').cat.codes
df['hopelessness']=df['hopelessness'].astype('category').cat.codes
df['Disorder']=df['Disorder'].astype('category').cat.codes

print(df.head())


X = df[['feeling.nervous', 'panic','breathing.rapidly', 'sweating','anger','over.react']]
y = df['Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.head())
print(y_train.head())
print(y.unique())

svm_model = SVC()
svm_model.fit(X_train, y_train)
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
logreg_accuracy = accuracy_score(y_test, logreg_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(svm_accuracy)
print(logreg_accuracy)
print(svm_accuracy)

