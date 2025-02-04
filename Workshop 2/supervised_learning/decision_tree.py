'''
Workshop #2.2 - AI Foundations

Welcome again! This is the code used in the second workshop. Here, you will see another 
Machine Learning Model called Decision Tree model.

As always, feel free to play around with this code :)

Use this to learn and remember to always have fun!
'''
#First, we import the libraries we will be using for today's project:
#1. scikit-learn - Used to create our Machine Learning Model
#2. Pandas - Used for Data Handling
#3. Matplotlib - Used for Data Visualization
from sklearn.preprocessing import LabelEncoder #Importing Preprocessing Tools
from sklearn.model_selection import train_test_split #Tool to Split the Data into Training and Testing 
from sklearn.tree import DecisionTreeClassifier #Importing our Decision Tree model
from sklearn.tree import plot_tree #To visualize the our Tree Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_score #Used to see data results

#Remeber to Install the libraries using "pip install <library>" to be able to access them

#1) IMPORTING THE DATA ------------------------------------------------------------------
df = pd.read_csv('Workshop 2/supervised_learning/data.csv') #df stands for DataFrame

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

#2.3) Spliting the Data into Training and Testing Data
x = df.drop(columns= ['Churn']) #Features
y = df['Churn'] #Target Column

#Now, since we have a large DataSet, we can split it into a training dataset and a testing dataset, and
#for that, the parameter "test_size" will take, in this case, 30% of our data to be testing and 70% for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 42)

#3) TRAINING DATA -----------------------------------------------------------------------

clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 4, min_samples_split= 2,min_samples_leaf= 2,
                            max_features= df.shape[1]) #You can try adding or adjusting the parameters to make a better model!

clf.fit(x_train, y_train) #Train

y_predicted = clf.predict(x_test) #Predict Testing Data

#4) PREDICTING DATA ---------------------------------------------------------------------

accuracy = precision_score(y_test, y_predicted) #Compare predictions with the actual data

print(f"Accuracy of our Decision Tree model: {accuracy * 100:.2f}%")
print("-" * 100)

#5) PLOTTING RESULTS --------------------------------------------------------------------

plt.figure(figsize=(25,15))
plot_tree(clf, filled= True, feature_names= df.columns, class_names= ['No Churn', 'Churn'], 
          rounded=True, proportion=False)

plt.title("Decision Tree Classifier Visualization")
plt.show()