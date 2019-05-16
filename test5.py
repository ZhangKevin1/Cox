import pandas as pd
import statsmodels.formula.api as smf
import math
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('E:\\flchain.csv')
del data["chapter"]
data = data.dropna()
data["lam"] = data["lambda"]
data["female"] = (data["sex"] == "F").astype(int)
data["year"] = data["sample.yr"] - min(data["sample.yr"])
print(data.head())
titleList = data.columns.values.tolist()
print(titleList)
record_keys = ['age', 'female', 'creatinine', 'year', 'death', 'futime']
x_keys = ['age', 'female', 'creatinine', 'year']
y_keys = ['futime']
for a in titleList:
    if a not in record_keys:
        del data[a]
print(data.head())
X = data[x_keys]
Y = data[y_keys]
status = data["death"].values

seed = 7
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# 拟合XGBoost模型
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
print(model.summary())

# 对测试集做预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
