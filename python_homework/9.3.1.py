class Person:
    def __init__(individual,name,age,gender):
        individual.name=name
        individual.age=age
        individual.gender=gender
    def personinfo(individual):
        print(f"姓名: {individual.name},年龄:{individual.age},性别:{individual.gender}")
p1 = Person("交小西",21,"男")
p1.personinfo()
