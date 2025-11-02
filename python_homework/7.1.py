import random
import statistics

file="data_input.txt"
with open(file,"w") as f:
    for i in range(10):
        num1=random.randint(1,100)
        num2=random.randint(1,100)
        num3=random.randint(1,100)
        f.write(str(num1)+","+str(num2)+","+str(num3)+"\n")

col = []
with open(file,"r") as f:
    for row in f:
        v=row.strip().split(",")
        col.append(int(v[1]))

max_col = max(col)
min_col = min(col)
avg_col = sum(col)/len(col)
med_col = statistics.median(col)

print("最大值:",max_col)
print("最小值:",min_col)
print("平均值:",avg_col)
print("中位数:",med_col)
