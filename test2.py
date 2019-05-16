x1 = [149.5, 162.5, 162.7, 162.2, 156.5, 156.1, 172.0, 173.2, 159.5, 157.7]
x2 = [69.5, 77.0, 78.5, 87.5, 74.5, 74.5, 76.5, 81.5, 74.5, 79.0]
sum1=0
sum2=0
count1=0
count2 = 0
a=0

for x in x1:
    sum1 = sum1 + x
    count1 = count1 +1
avg1 = sum1/count1
print(avg1)

for x in x2:
    sum2 = sum2 + x
    count2 = count2 +1
avg2 = sum2/count2
print(avg2)

for x in x1:
    a = a+(x-161.2)*(x-161.2)
result = a/count1
print(result)
