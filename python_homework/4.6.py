'''
if __name__=="__main__":
    list=list(range(1000))
    for idx in range(len(list)):
        if list[idx]%2==1:
            list.pop(idx) 
            #删除元素时列表元素个数减少,整体前移,后面下标会超出列表现有范围
'''
if __name__=="__main__":
    list=list(range(1000))
    for idx in range(999,0,-1):
        if list[idx]%2==1:
            list.pop(list[idx])
