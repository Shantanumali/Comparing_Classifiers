import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('suv_data.csv')

df = df.drop(['User ID'], axis=1,inplace=False)
sex = pd.get_dummies(df['Gender'],drop_first=True)
df = pd.concat([df, sex], axis = 1)
df = df.drop(['Gender'],axis = 1)
X = df.drop(['Purchased'], axis=1)
Y = df['Purchased']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=10)

names = ["LogisticRegression", "Nearest_Neighbors", "Decision_Tree", "Random_Forest", "Naive_Bayes"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=40),
    GaussianNB()]

scores = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    scores.append(score)

for i, j in zip(names, scores):
    print(i,":","{:.2f}".format(j*100),"%")

df = pd.DataFrame()
df['name'] = names
df['score'] = scores
sns.barplot(y="name", x="score", data=df)
plt.show()