from math import sqrt
from anothermethod import *

record = {}
comparerecord = {}

# 欧几里德距离
def sim_distance(prefs, p1, p2):
    si = {}
    temp = {}
    temp[p1] = {}
    temp[p2] = {}
    for user in prefs[p1]:
        si[user] = 1
        temp[p1][user] = 1
    for user in prefs[p2]:
        si[user] = 1
        temp[p2][user] = 1

    for user in si:
        if user not in temp[p1]:
            temp[p1][user] = 0
        if user not in temp[p2]:
            temp[p2][user] = 0

    if len(si) == 0:
        return 0
    # cal the distance
    sum_of_sqr = sum([pow(temp[p1][user]-temp[p2][user], 2) for user in si])
    return 1/(1+sqrt(sum_of_sqr))


# 皮尔逊相关度
def sim_pearson(prefs, p1, p2):
    si = {}
    temp = {}
    temp[p1] = {}
    temp[p2] = {}
    for user in prefs[p1]:
        si[user] = 1
        temp[p1][user] = 1
    for user in prefs[p2]:
        si[user] = 1
        temp[p2][user] = 1
    for user in si:
        if user not in temp[p1]:
            temp[p1][user] = 0
        if user not in temp[p2]:
            temp[p2][user] = 0
    if len(si) == 0:
        return 0
    sum1 = sum([temp[p1][user] for user in si])
    sum2 = sum([temp[p2][user] for user in si])
    avg1 = sum1/len(si)
    avg2 = sum2/len(si)
    sum1sq = sum([pow((temp[p1][user]-avg1), 2) for user in si])
    sum2sq = sum([pow((temp[p2][user]-avg2), 2) for user in si])
    psum = sum([(temp[p1][user]-avg1)*(temp[p2][user]-avg2) for user in si])
    den = sqrt(sum1sq*sum2sq)
    if den == 0:
        return 0
    return psum/den


# 余弦相似度
def sim_cosin(prefs, p1, p2):
    si = {}
    temp = {}
    temp[p1] = {}
    temp[p2] = {}
    for user in prefs[p1]:
        si[user] = 1
        temp[p1][user] = 1
    for user in prefs[p2]:
        si[user] = 1
        temp[p2][user] = 1
    for user in si:
        if user not in temp[p1]:
            temp[p1][user] = 0
        if user not in temp[p2]:
            temp[p2][user] = 0
    if len(si) == 0:
        return 0
    sum_p1ANDp2 = sum([temp[p1][user]*temp[p2][user] for user in si])
    p1_distance = sqrt(sum([pow(temp[p1][user], 2) for user in si]))
    p2_distance = sqrt(sum([pow(temp[p2][user], 2) for user in si]))
    result = sum_p1ANDp2/(p1_distance*p2_distance)
    return result


# 物品与人员互相调换
def transformprefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # 人、物对调
            result[item][person] = prefs[person][item]
    return result


# 最相关的N个item
def topmatchs(prefs, item, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, item, other), other) for other in prefs if other != item]
    scores.sort(reverse=True)
    print('*********与套餐 %s 相似度最高的10个是**************' % item)
    print(scores[0:n])
    return scores[0:n]


# 构建物品比较数据集合,即，每个item有n个最相关的其它item
def calculatesimilaryitems(prefs, n=10, similarity=sim_cosin):
    itemsim = {}
    itemprefs = transformprefs(prefs)
    c = 0
    for item in itemprefs:
        c += 1.0
        if c % 100 == 0:
            print('%d/%d' % (c, len(itemprefs)))
        scores = topmatchs(itemprefs, item, n, similarity)
        itemsim[item] = scores
    print("**************itemsim***************")
    print(itemsim)
    print('*********计算相似度完成！*********')
    return itemsim


# 获取推荐item
def getrecommendations(prefs, itemsim, user, state):
    if state == 0:
        rank = dict()
        action_item = prefs[user]  # 用户user产生过行为的item和评分
        for item, ratio in action_item.items():
            for item2, similarity in itemsim[item]:
                if item2 in action_item.keys():
                    continue
                rank.setdefault(item2, 0)
                rank[item2] += similarity
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:10])
    else:
        rank = dict()
        action_item = prefs[user]  # 用户user产生过行为的item和评分
        for item, score in action_item.items():
            for j, wj in sorted(itemsim[item].items(), key=lambda x: x[1], reverse=True)[0:10]:
                if j in action_item.keys():
                    continue
                rank.setdefault(j, 0)
                rank[j] += score * wj
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:10])



def initial():
    # 取输入文件的数据，存入record
    count = 0
    with open('D:\ItemBased模型开发/ncf.train', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            count += 1
            lineSplit = line.split(',')
            user = lineSplit[0]
            item = lineSplit[1]
            record.setdefault(user, {})
            record[user][item] = 1
            if count%100000 == 0:
                print(count)
        print(record)
    print(count)

    # 取对照文件中的数据，存入comparerecord，后面进行匹配计算
    with open('D:\ItemBased模型开发/users', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            lineSplit = line.split(',')
            user = lineSplit[0]
            item = lineSplit[1]
            comparerecord[user] = item
        print(comparerecord)


if __name__ == '__main__':
    initial()
    # 根据state选择调用上面三种相似度计算方法，还是调用anothermethod.py里的同现矩阵相似度计算方法
    state = 1
    if state == 0:
        itemsim = calculatesimilaryitems(record, 10, sim_cosin)
    else:
        itemsim = sim_w(record)
    for person in record:
        rankings = getrecommendations(record, itemsim, person, state)
        print('***********为用户%s推荐的套餐为************' % person)
        print(rankings)
        with open('D:\ItemBased模型开发/result_train', 'a') as f:
            f.write(person)
            f.write(str(rankings))
            f.write('\n')

        # 计算是否匹配，得到平均数
        match = 0
        for item in rankings:
            if item == comparerecord[person]:
                match = 1
                break

        with open('D:\ItemBased模型开发/match', 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            f.write(person)
            f.write(',')
            f.write(str(match))
            f.write('\n')

