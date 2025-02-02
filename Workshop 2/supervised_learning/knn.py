'''
Workshop #2 - AI Foundations

Welcome again! This is code used in the second workshop. Here, you will see one of most 
used Machine Learning Models called K-Nearest-Neighbors.

As always, feel free to play around with this code :)

Use this to learn and remember to always have fun!
'''
#First, we import the libraries we will be using for today's project:
#1. scikit-learn - Used to create our Machine Learning Model
#2. Pandas - Used for Data Handling
#3. Matplotlib - Used for Data Visualization
from sklearn.preprocessing import MinMaxScaler, LabelEncoder #Importing Preprocessing Tools
from sklearn.neighbors import KNeighborsClassifier #Importing Model
import pandas as pd
import matplotlib.pyplot as plt 

#Remeber to Install the libraries using "pip install <library>" to be able to access them

#1) IMPORTING THE DATA ------------------------------------------------------------------
df = pd.read_csv('Workshop 2/supervised_learning/data.csv')

print("Rows and Columns:", df.shape) #Get total number of rows and columns
print("-" * 100)

#2) DATA HANDLING  ----------------------------------------------------------------------
print("Columns of Our Data:", df.columns)

print("-" * 100)
#2.1) Removing Unnecessary Data
columns_removed = ['customerID', 'Gender', 'SeniorCitizen', 
                   'Partner', 'Dependents', 'InternetService', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod'] #These are the columns that we want to remove

df.drop(columns_removed, axis= 1, inplace= True)

print("New Rows and Columns:", df.shape) #Get total number of rows and columns
print("Columns of Our Data after Dropping Columns:", df.columns)
print("-" * 100)

#2.2) Enconding Data
categorical_features = [] 

#We need to get the columns that have words instead of numbers, because we can't work with words like "Yes" or "No" in a ML Model
for column in df:
    if df[column].dtype == object:
        categorical_features.append(column)

print("Categorical Columns:", categorical_features)

#After getting the columns with words, now we can encode them by turning these words (Yes/No) into numbers (1/0)
label_encoder = LabelEncoder()

for column in categorical_features:
    df[column] = label_encoder.fit_transform(df[column])

print("Columns after Enconding:", df[categorical_features])
print("-" * 100)

#2.3) Scaling Our Data


#3) Training the Data -------------------------------------------------------------------

#4) Predicting the Data  ----------------------------------------------------------------