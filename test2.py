import pandas as pd
import statsmodels.formula.api as smf
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pylab as plt


record_keys = []


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


# 测试，得到预测准确率
def predict(record, params, state, S0):
    # 在训练集中是否存在与预测时间相同时刻，是为1，否为0
    # match = 0
    # for time in record:
    #     if matchtime < time <= testTime:
    #         matchtime = time

    # 计算每条数据的生存率，match为在训练集中是否存在与预测时间相同时刻，若为0，则不对该条数据进行预测
    for time in record:
        match = 0
        for time2 in S0:
            if time == time2:
                match = 1
                S01value = S0[time]
        for value in record[time]:
            if match == 1:
                temp = 0
                for x in value:
                    if x is not state:
                        temp = temp + value[x] * params[x]
                b = math.exp(temp)
                S = math.pow(S01value, b)
                value['S'] = S
                if time == 34:
                    print(str(time)+ ":"+ str(S01value) + ":" + str(S))
            else:
                value['S'] = 'Not Match Time'


    #输出结果保存到文件中
    outputFile = 'E:\\output.txt'
    with open(outputFile, 'w') as f:
        f.write(state + '\t')
        f.write('time' + '\t')
        f.write('Survival Rate\n')
        for time in record:
            for value in record[time]:
                f.write(str(value[state]))
                f.write('\t')
                f.write(str(time))
                f.write('\t')
                f.write(str(value['S']))
                f.write('\n')

    print("结果输出至文件：" + outputFile)







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

    # testTime = 50
    print("节点一"+str(S0[34]))
    predict(testRecord, params, state, S0)






