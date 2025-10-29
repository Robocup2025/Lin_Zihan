s=0
print("满足条件的三位数有:")
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            if(i!=j and j!=k and k!=i):
                a=i*100+j*10+k
                s+=1
                print(a)
print("有%d个数"%(s))