'''
Workshop #1 - Python Review

Welcome! This is the code used in the first workshop. Here, we discuss If Statements, For Loops, While Loops,
Functions, Basics of OOP (Object-Oriented-Programming) and Libraries!

Feel free to play around with this code :)

Use this to learn and remember to always have fun!
'''
#--------------------------------
#1) If Staments
print("1) If Statement:")

condition1 = True
condition2 = False
condition3 = False
condition4 = True

if condition1:
    print("Condition1 is True!")
else:
    print("Condition1 is False!")

if condition2:
    print("Condition2 is True!")
elif condition3:
    print("Condition3 is True!")
elif condition4:
    print("Condition4 is True!")
else:
    print("All conditions are False!")

print("")
#--------------------------------
#2) For Loops
print("2) For Loops:")

myArray = [1, 2, 3, 4, 5]

for element in myArray:
    print(element)

size = len(myArray)
for index in range(size):
    print("Element at index", index, "is", myArray[index])

print("")
#--------------------------------
#3) While Loops
print("3) While Loops:")

myArray = [2, 4, 5, 8, 10, 12, 15]

index = 0
sum = 0
while sum < 20:

    print("Sum:", sum)
    print("Index:", index)

    sum += myArray[index]
    index += 1

print("")
#--------------------------------
#4) Functions
print("4) Functions:")

def myFunction_Sum(Parameter_1, Parameter_2):
    return Parameter_1 + Parameter_2

result = myFunction_Sum(5, 10)

print("Result:", result)

print("")
#--------------------------------
#5) Object-Oriented-Programming
print("5) Object-Oriented-Prog (OOP)")

class FIU_student:

    def __init__(self, name, PID): #Constructor
        self.name = name
        self.PID = PID

    age = int() #Attribute 
    
    def getName(self): #Method 1
        print("My name is:", self.name) 
    
    def getPID(self): #Method 2
        print("My Panther ID is:", self.PID)

#Defining Object student of class FIU_student 
student = FIU_student("Peter", "1234567") 
student.age = 20

student.getName()
student.getPID()

print("I am", student.age, "years old")

print("")

#--------------------------------
#6) Libraries
from sklearn.neighbors import KNeighborsClassifier as knc

#To install the library, run the following command in your terminal:
# "pip install scikit-learn" OR "pip3 install scikit-learn"
# To check IF you have installed, you can run the following command:
# "pip show scikit-learn"

knn = knc(n_neighbors= 3) #Initialize Object

'''
x_data = []
y_data = []

#Access method "fit" to train the data
knn.fit(x_data, y_data) 

#Access method "predict" to predict the testing data
x_testing = []
y_prediction = knn.predict(x_testing) 
'''
#If you are having issues installing the libraries, you can use a 
#virtual environment by typing the following commands:

#1) "python3 -m venv myenv"
#2.1) "source myenv/bin/activate" - MacOS/Linux - activates the virtual environment
#2.2) "myenv\Scripts\activate" - Windows - activates the virtual environment
#3) "pip install scikit-learn" - Install scikit-learn in the virtual environment
