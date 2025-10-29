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
    def __str__(self):
        return f"姓名:{self.name},年龄:{self.age},性别:{self.gender},学院:{self.college},班级:{self.classes}"
p=[Student("交小西",21,"男","钱院","钱班"),Student("交小招",20,"女","文治书院","智感2402"),Student("交小智",22,"未知","仲英书院","自动化2407")]
for i in p:
    print(i)
