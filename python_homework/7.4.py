li1=[]
li2=[]
with open("test.txt","r") as f:
    for row in f:
        li1.append(row)
with open("copy_test.txt","r") as f:
    for row in f:
        li2.append(row)
for i in range(30):
    if(li1[i]!=li2[i]):
        print(i)
