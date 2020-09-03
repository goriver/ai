import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_df = pd.read_cvs("../dataset/mnist_train.csv")
test_df = pd.read_cvs("../dataset/mnist_test.csv")

train_data = train_df.values
# 8:2 -> 학습데이터:검증데이터 
X_train, X_test, Y_train, Y_test = train_test_split(train_data[0:, 1:], train_data[0:, 0], test_size=0.2, random_state=0)

# RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=500)
rdc = model1.fit(X_train, Y_train)
output1 = rdc.predict(X_test)

# GradientBoostingClassifier

# MLPClassifier