class Person:
    def __init__(individual,name,age,gender):
        individual.name=name
        individual.age=age
        individual.gender=gender

class Student(Person):
    def __init__(individual,name,age,gender,college,classes):
        super().__init__(name,age,gender)
        individual.college=college
        individual.classes=classes
    def personinfo(individual):
        print(f"姓名:{individual.name},年龄:{individual.age},性别:{individual.gender},学院:{individual.college},班级:{individual.classes}")
p2=Student("交小西",21,"男","钱院","钱班")
p2.personinfo()
