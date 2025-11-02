import random
import shutil

file="test.txt"
with open(file,"w") as f:
    for i in range(30): #指定生成30行内容
        character=chr(random.randint(32, 126))
        f.write(character+"\n")

shutil.copy("test.txt","copy_test.txt")

