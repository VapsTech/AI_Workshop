'''
Workshop #3 - Linear Regression & Neural Networks

Welcome again! This is the code used in the third workshop. Here, you will see one of
most important Machine Learning model using statistics called Linear Regression!

As always, feel free to play around with this code :)

Use this to learn and remember to always have fun!
'''
#If you are on MacOS, you can try creating a virtual environmnet 
#to avoid installation issues with the libraries
#Follow the Commands:
#1) python3 -m venv .myenv
#2) source .myenv/bin/activate

#Libraries to Install:
#1) pip install numpy
#2) pip install pandas
#3) pip install scikit-learn
#4) pip install matplotlib
#5) pip install seaborn

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#1) IMPORTING DATA ----------------------------------------------------------------------
data = pd.read_csv('Workshop 3/data/data.csv')

df = pd.DataFrame(data)

#2) DATA HANDLING -----------------------------------------------------------------------

# fig, axs = plt.subplots(2,3, figsize = (10,5))
# plt1 = sns.boxplot(df['price'], ax = axs[0,0])
# plt2 = sns.boxplot(df['area'], ax = axs[0,1])
# plt3 = sns.boxplot(df['bedrooms'], ax = axs[0,2])
# plt1 = sns.boxplot(df['bathrooms'], ax = axs[1,0])
# plt2 = sns.boxplot(df['stories'], ax = axs[1,1])
# plt3 = sns.boxplot(df['parking'], ax = axs[1,2])

# plt.tight_layout()
# plt.show()

#2.1) Removing Outliers
#Explanation: Since we are trying to build a line, outliers in the data can directly affect the
#calucations for the line, so we need to remove them first

print("Original Shape:", df.shape)

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

outlier = []

for index, price in df['price'].items():
    if price >= (Q3 + 1.5*IQR) or price <= (Q1 - 1.5*IQR):
        outlier.append(index)

Q1 = df['area'].quantile(0.25)
Q3 = df['area'].quantile(0.75)
IQR = Q3 - Q1

for index, area in df['area'].items():
    if area >= (Q3 + 1.5*IQR) or area <= (Q1 - 1.5*IQR):
        outlier.append(index)

df.drop(outlier, axis= 0, inplace= True)

print("Shape after Removing the Outliers:", df.shape)

#2.2) Removing Unnecessary Columns
#Explanation: Sometimes, some columns in our Data may not place a huge difference in the result,
#and this allows us to remove them to avoid more work or bad results

#Here, I decided to remove just the furnishingstatus column
df.drop(columns= ['furnishingstatus'], axis= 1, inplace= True)
print("Shape after removing unnecessary Column(s):", df.shape)

#2.3) Encoding Categorical Features
#Explanation: In a Machine Learning Model, we can't work with categirical data (words like "yes" and "no"), 
# that's why we need to turn them into numbers (like 1 and 0)

categorical_columns = []

#Let's first get all the columns that have strings instead of numbers
for column in df:
    if df[column].dtype == object:
        categorical_columns.append(column)

le = LabelEncoder()

for categorical in categorical_columns:
    df[categorical] = le.fit_transform(df[categorical]) #Turning Yes/No into 1/0

#2.4) Splitting the Data into Training Data and Testing Data

#In this case, 80% of the whole data will be training data and 20% will be testing data
df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 100)

#2.5) Scaling Our Data
#Explanation: some features have different ranges of values like one going from 100_000 to 1_000_000, and 
#another feature going from 0 to 1 only! So we adjust these values to put them in the SAME scale! Otherwise,
#our model can be very biased

scaler = MinMaxScaler()

# Apply scaler() to all the columns except for the 'yes-no' (1/0)
scale_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[scale_columns] = scaler.fit_transform(df_train[scale_columns])
df_test[scale_columns] = scaler.fit_transform(df_test[scale_columns])

#3) TRAINING DATA -----------------------------------------------------------------------

model = LinearRegression()

x_train = df_train.drop(columns= 'price')
y_train = df_train['price']

model.fit(x_train, y_train)

#4) PREDICTING DATA ---------------------------------------------------------------------

x_test = df_test.drop(columns= 'price')
y_test = df_test['price']

y_prediction = model.predict(x_test)

#5) RESULTS & EVALUATION OF OUR MODEL ---------------------------------------------------
accuracy = r2_score(y_test, y_prediction)
print("r^2 Score =", accuracy)

mse = mean_squared_error(y_test, y_prediction)
print("Mean Squared Error =", mse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_prediction, alpha=0.5, color='blue', label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Perfect Fit Line")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()