with open("test.txt","r+") as f:
    old=f.read()
    f.seek(0)
    f.write("python"+old)
    f.seek(0,2)
    f.write("python")
