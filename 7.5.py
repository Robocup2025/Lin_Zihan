from pathlib import Path
import os
import random

Path("test").mkdir(exist_ok=True)
for i in range(30): #指定自建30个文件
    (Path("test")/f"{i}.txt").touch()
    with open(Path("test")/f"{i}.txt","w") as f:
        for j in range(20): #指定生成20行内容
            character=chr(random.randint(32, 126))
            f.write(character+"\n")

for i in range(30):
    old=Path("test")/f"{i}.txt"
    new=Path("test")/f"{i}-python.txt"
    os.rename(old,new)
    with open(new,"r+") as f:
        raw=f.readlines()
        f.seek(0)
        f.writelines(line.strip()+"-python\n"for line in raw)
        f.truncate()