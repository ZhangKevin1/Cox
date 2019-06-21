import statsmodels.api as sm
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pylab as plt


record_keys = []

# x = ['SeniorCitizen', 'Partner', 'Dependents',  'PhoneService',
#          'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
#          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
#          'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
#     record_keys = x
#     y = 'tenure'
#     state = 'Churn'

def initial(file, x, y, state):
    data = pd.read_csv(file)
    data = data.dropna()
    data["gender"] = (data["gender"] == "Male").astype(int)
    data["Partner"] = (data["Partner"] == "Yes").astype(int)
    data["Dependents"] = (data["Dependents"] == "Yes").astype(int)
    data["PhoneService"] = (data["PhoneService"] == "Yes").astype(int)
    data["MultipleLines"] = data['MultipleLines'].map({'No phone service': 0, 'No': 1, 'Yes': 2})
    data["InternetService"] = data['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
    data["TechSupport"] = data['TechSupport'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    data["StreamingTV"] = data['StreamingTV'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    data["Contract"] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    data["PaperlessBilling"] = (data["PaperlessBilling"] == "Yes").astype(int)
    data["PaymentMethod"] = data['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    data["Churn"] = (data["Churn"] == "Yes").astype(int)
    print(data.head())
    print("最开始的数量：", len(data))

    # 切分为训练集和测试集
    titleList = data.columns.values.tolist()
    print(titleList)
    x_keys = []
    for key in x:
        x_keys.append(key)
    x_keys.append(y)
    x_keys.append(state)
    y_keys = []
    for a in titleList:
        if a not in x_keys:
            del data[a]
    print(data.head())
    X = data[x_keys]
    Y = data[y_keys]
    seed = 7
    test_size = 0.4
    trainData, testData, ab, cd = train_test_split(X, Y, test_size=test_size, random_state=seed)

    print("切分后训练集data：", len(trainData))
    print("切分后测试集data：", len(testData))

    status = trainData[state].values

    sentence = y + "~"
    count = 0
    for key in record_keys:
        if count == 0:
            sentence = sentence + key
        else:
            sentence = sentence + "+" + key
        count = count + 1
    print(sentence)

    # mod = smf.phreg("futime ~ age + female + creatinine + "
    #                 "  + year",
    #                 trainData, status=status, ties="efron")

    mod = smf.phreg(sentence, trainData, status=status, ties="efron")
    rslt = mod.fit()
    print(rslt.summary())
    # 得到h(t|X)=h0(t)exp(X^T*B)的协变量参数B
    params = {}
    i = 0
    while i < len(record_keys):
        params[record_keys[i]] = rslt.params[i]
        i = i + 1
    print(params)
    return trainData, testData, params




# 将data数据存入list record中
# 构建存活时间与其他值的键值对
def getRecord(data, y, state):
    record = {}
    for index, row in data.iterrows():
        oneRecord = {}
        futime = row[y]
        death = row[state]
        for x in record_keys:
            oneRecord[x] = row[x]
        oneRecord[state] = death
        record.setdefault(futime, []).append(oneRecord)
    print(record)
    num = 0
    for time in record:
        num = num + len(record[time])
    print("记录条数：", num)
    return record

# 用Breslow法估计出基准生存函数S0(ti)
# h0为基准风险函数，H0为基准累积风险函数，S0为基准生存率
def getS0(record, params, state):
    h0 = {}
    H0 = {}
    for time in record:
        a = len(record[time])
        for value in record[time]:
            if value[state] is 0:
                a = a - 1
        sumb = 0
        for time2 in record:
            if time <= time2:
                for value in record[time2]:
                    temp = 0
                    for x in value:
                        if x is not state:
                            temp = temp + value[x] * params[x]
                    b = math.exp(temp)
                    sumb = sumb + b
        h0[time] = a / sumb
    print(h0)
    for time in h0:
        temp = 0
        for time2 in h0:
            if time2 < time:
                temp = temp + h0[time2]
        H0[time] = temp
    print(H0)

    # 得到S0
    S0 = {}
    for time in H0:
        S0[time] = math.exp(H0[time] * (-1))
    print(S0)
    return S0


# 获取评估值
def  getEvaluation(record, state, testTime):
    smaller = 0
    censor = 0
    number = 0
    for time in record:
        if time <= testTime:
            for value in record[time]:
                if value[state] is 0:
                    censor = censor + 1
                else:
                    smaller = smaller + 1
        else:
            censor = censor + len(record[time])
        number = number + len(record[time])

    print("死亡数：", smaller)
    print("删失数：", censor)
    print("总数：", number)
    evaluation = smaller / number
    print("评估值为：", evaluation)

    # 获得与预测时间最相近的训练集时刻
    matchtime = 0
    for time in record:
        if matchtime < time <= testTime:
            matchtime = time
    return evaluation, matchtime

# 测试，得到预测准确率
def predict(record, params, state, S0, evaluation, matchtime):
    deathmatch = 0
    notdeathmatch = 0
    testdeath_realnot = 0
    testnotdeath_realdeath = 0

    TrueList = []
    Predictlist = []

    # 取得测试时间下的基准生存率
    S0Test = S0[matchtime]

    print("matchTime为：", matchtime)

    smallerMatchTimeNum = 0
    largerMatchTimeNum = 0
    for time in record:
        if time < matchtime:
            smallerMatchTimeNum = smallerMatchTimeNum + len(record[time])
        else:
            largerMatchTimeNum = largerMatchTimeNum + len(record[time])
    print("小于预测时间的数量：", smallerMatchTimeNum)
    print("大于预测时间的数量：", largerMatchTimeNum)

    # 计算每条数据的生存率，与评估值对比，预测是否发生流失
    for time in record:
        for value in record[time]:
            temp = 0
            for x in value:
                if x is not state:
                    temp = temp + value[x] * params[x]
            b = math.exp(temp)
            S = math.pow(S0Test, b)
            # 将实际值与预测生存率分别存入列表
            TrueList.append(value[state])
            Predictlist.append(1-S)

            # 生存率小于等于评估值，则预测为流失，否则为生存
            if S <= evaluation:
                predict = 1
            else:
                predict = 0
            # 若该条数据的时间大于测试时间，则说明该数据实际存活
            # 若小于测试时间，要与death字段比较，看是否为删失数据
            if time > matchtime:
                real = 0
            elif value[state] is 0.0:
                real = 0
            else:
                real = 1
            if predict is 1 and real is 1:
                deathmatch = deathmatch + 1
            elif predict is 1 and real is 0:
                testdeath_realnot = testdeath_realnot + 1
            elif predict is 0 and real is 1:
                testnotdeath_realdeath = testnotdeath_realdeath + 1
            else:
                notdeathmatch = notdeathmatch + 1

    print("流失匹配：", deathmatch)
    print("非流失匹配：", notdeathmatch)
    print("预测流失实际非流失：", testdeath_realnot)
    print("预测非流失实际流失：", testnotdeath_realdeath)
    print("预测准确率：",
          (deathmatch + notdeathmatch) / (deathmatch + notdeathmatch + testnotdeath_realdeath + testdeath_realnot))

    # 绘制Roc曲线
    fpr, tpr, thresholds = metrics.roc_curve(TrueList, Predictlist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.show()




# initial(file) return trainData,testData,params
# getRecord(data, y, state) return record
# getS0(record, params) return S0
# getEvaluation(record, testTime) return evaluation
# predict(record, params, S0, evaluation, testTime)
if __name__ == '__main__':
    file = "E:\\Telcom_data.csv"
    x = ['SeniorCitizen', 'Partner', 'Dependents',  'PhoneService',
         'MultipleLines', 'InternetService', 'TechSupport', 'StreamingTV',
         'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']
    record_keys = x
    y = 'tenure'
    state = 'Churn'
    trainData, testData, params = initial(file, x, y, state)
    trainRecord = getRecord(trainData, y, state)
    testRecord = getRecord(testData, y, state)
    S0 = getS0(trainRecord, params, state)

    testTime = 50
    evaluation, matchTime = getEvaluation(trainRecord, state, testTime)
    predict(testRecord, params, state, S0, evaluation, matchTime)






