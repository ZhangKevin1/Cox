import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

data = pd.read_csv("E:\\flchain.csv")
data["female"] = (data["sex"] == "F").astype(int)
data["year"] = data["sample.yr"] - min(data["sample.yr"])
data = data[['female', 'creatinine', 'kappa', 'lambda', 'year']]
print(data.head())
print(data.corr())
# method='pearson' ，默认为*Pearson*相关系数

plt.figure(figsize=(12,9))
sns.set(font_scale=1.25)#解决seaborn不能显示中文的问题
# 使用热地图(heat map)更直观地展示系数矩阵情况
# vmax设定热地图色块的最大区分值
# square设定图片为正方形与否
# annot设定是否显示每个色块的系数值
print('asdas')
sns.heatmap(data.corr(),vmax=1,square=True,annot=True)
plt.show()
print('asaf')
