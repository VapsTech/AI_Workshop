'''
Workshop #2.1 - AI Foundations

Welcome again! This is the code used in the second workshop. Here, you will see one of most 
used Machine Learning Models called K-Nearest-Neighbors.

As always, feel free to play around with this code :)

Use this to learn and remember to always have fun!
'''
#First, we import the libraries we will be using for today's project:
#1. scikit-learn - Used to create our Machine Learning Model
#2. Pandas - Used for Data Handling
#3. Matplotlib - Used for Data Visualization
from sklearn.preprocessing import MinMaxScaler, LabelEncoder #Importing Preprocessing Tools
from sklearn.model_selection import train_test_split #Tool to Split the Data into Training and Testing 
from sklearn.neighbors import KNeighborsClassifier #Importing our Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Used to see data results

#Remeber to Install the libraries using "pip install <library>" to be able to access them

#1) IMPORTING THE DATA ------------------------------------------------------------------
df = pd.read_csv('') #df stands for DataFrame

print("Rows and Columns:", df.shape) #Get total number of rows and columns
print("-" * 100)

#2) DATA HANDLING  ----------------------------------------------------------------------
print("Columns of Our Data:", df.columns)

print("-" * 100)
#2.1) Removing Unnecessary Data


print("New Rows and Columns:", df.shape) #Get total number of rows and columns
print("Columns of Our Data after Dropping Columns:", df.columns)
print("-" * 100)

#2.2) Enconding Data


#We need to get the columns that have words instead of numbers, because we can't work with words like "Yes" or "No" in a ML Model




#After getting the columns with words, now we can encode them by turning these words (Yes/No) into numbers (1/0)




#2.3) Scaling Our Data
#We need to scale our data to adjust proportionally their values to a certain common range (Like from 0 to 1)
#so no feature/column with different range can bias our model




#2.4) Spliting the Data into Training and Testing Data


#Now, since we have a large DataSet, we can split it into a training dataset and a testing dataset, and
#for that, the parameter "test_size" will take, in this case, 30% of our data to be testing and 70% for training

#3) TRAINING DATA -------------------------------------------------------------------



#4) PREDICTING DATA  ----------------------------------------------------------------



# Get Accuracy of our Model
#points = 0
#size = len(y_test)
#for index in range(size):
#    if y_test.iloc[index] == y_predict[index]: #If we match the same value for the row, we increase our points
#        points += 1

#Divide the number of values we predicted correctly with the total number of values in the test
# accuracy = points / size 

# print(f"Accuracy of our KNN model: {accuracy * 100:.2f}%")
# print("-" * 100)

#5) PLOTTING OUR RESULTS -----------------------------------------------------------------

# Compute confusion matrix
#cm = confusion_matrix(y_test, y_predict) #Here, we use the actual values (y_test) and the values we predicted (y_predict)

# Display confusion matrix
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
#disp.plot(cmap= "Purples")
#plt.title("Confusion Matrix of KNN Model")
#plt.show()
