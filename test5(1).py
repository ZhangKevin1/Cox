import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
import math
from sklearn.model_selection import train_test_split


record_keys = ['age', 'female', 'creatinine', 'year']

def initial(file):
    data = pd.read_csv(file)
    del data["chapter"]
    data = data.dropna()
    data["lam"] = data["lambda"]
    data["female"] = (data["sex"] == "F").astype(int)
    data["year"] = data["sample.yr"] - min(data["sample.yr"])
    print(data.head())
    print("最开始的数量：", len(data))

    # 切分为训练集和测试集
    titleList = data.columns.values.tolist()
    print(titleList)
    x_keys = ['age', 'female', 'creatinine', 'year', 'death', 'futime']
    y_keys = ['age']
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

    status = trainData["death"].values

    mod = smf.phreg("futime ~ age + female + creatinine + "
                    "  + year",
                    trainData, status=status, ties="efron")
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
def getRecord(data):
    record = {}
    for index, row in data.iterrows():
        oneRecord = {}
        futime = row['futime']
        death = row['death']
        for x in record_keys:
            oneRecord[x] = row[x]
        oneRecord['death'] = death
        record.setdefault(futime, []).append(oneRecord)
    print(record)
    num = 0
    for time in record:
        num = num + len(record[time])
    print("记录条数：", num)
    return record

# 用Breslow法估计出基准生存函数S0(ti)
# h0为基准风险函数，H0为基准累积风险函数，S0为基准生存率
def getS0(record, params):
    h0 = {}
    H0 = {}
    for time in record:
        a = len(record[time])
        for value in record[time]:
            if value['death'] is 0:
                a = a - 1
        sumb = 0
        for time2 in record:
            if time <= time2:
                for value in record[time2]:
                    temp = 0
                    for x in value:
                        if x is not 'death':
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
def  getEvaluation(record, testTime):
    smaller = 0
    censor = 0
    number = 0
    for time in record:
        if time <= testTime:
            for value in record[time]:
                if value['death'] is 0:
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
    return evaluation

# 测试，得到预测准确率
def predict(record, params, S0, evaluation, testTime):
    deathmatch = 0
    notdeathmatch = 0
    testdeath_realnot = 0
    testnotdeath_realdeath = 0
    matchtime = 0

    # 取得测试时间下的基准生存率
    for time in record:
        if matchtime < time <= testTime:
            matchtime = time
    S0Test = S0[matchtime]

    print("matchTime为：", matchtime)

    smallerMatchTimeNum = 0
    largerMatchTimeNum = 0
    for time in record:
        if time < matchtime:
            smallerMatchTimeNum = smallerMatchTimeNum + 1
        else:
            largerMatchTimeNum = largerMatchTimeNum + 1
    print("小于预测时间的数量：", smallerMatchTimeNum)
    print("大于预测时间的数量：", largerMatchTimeNum)

    # 计算每条数据的生存率，与评估值对比，预测是否发生流失
    for time in record:
        for value in record[time]:
            temp = 0
            for x in value:
                if x is not 'death':
                    temp = temp + value[x] * params[x]
            b = math.exp(temp)
            S = math.pow(S0Test, b)
            # 生存率小于等于评估值，则预测为流失，否则为生存
            if S <= evaluation:
                predict = 1
            else:
                predict = 0
            # 若该条数据的时间大于测试时间，则说明该数据实际存活
            # 若小于测试时间，要与death字段比较，看是否为删失数据
            if time > matchtime:
                real = 0
            elif value['death'] is 0.0:
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


# initial(file) return trainData,testData,params
# getRecord(data) return record
# getS0(record, params) return S0
# getEvaluation(record, testTime) return evaluation
# predict(record, params, S0, evaluation, testTime)
if __name__ == '__main__':
    file = "E:\\flchain.csv"
    trainData, testData, params = initial(file)
    trainRecord = getRecord(trainData)
    testRecord = getRecord(testData)
    S0 = getS0(trainRecord, params)

    testTime = 4000
    predict(testRecord, params, S0, getEvaluation(trainRecord, testTime), testTime)






