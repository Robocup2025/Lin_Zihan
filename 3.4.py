a=input("请输入正整数:")
s1=str(a)
l=len(s1)
s2=s1[-1:-l-1:-1]
if s2==s1:
    print("是回文数")
else:
    print("不是回文数")