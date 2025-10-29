import random
with open("data.txt","w") as file:
    for i in range(100000):
        num=random.randrange(1,101)
        file.write(str(num)+"\n")
        