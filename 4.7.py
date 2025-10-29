li=list(range(1,234))
k=0
l=len(li)
while l>1:
    k=(k+2)%l
    li.pop(k)
    l-=1
print(li[0])