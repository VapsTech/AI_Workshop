'''
Workshop #3 - Linear Regression & Neural Networks

Welcome again! This is the code used in the third workshop. Here, you will see how
we can use Neural Networks (Gradient Descent) in Deep Learning (Not Machine Learning anymore xD) 
to make extrodinary results.

For that, we will be using the same data from the Linear Regression model!

As always, feel free to play around with this code :)

Use this to Learn and remember to always have fun!
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('Workshop 3/data/data.csv')

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
df.drop(columns= ['furnishingstatus'], 
        axis= 1, inplace= True)

print("Shape after Removing the Outliers:", df.shape)

categorical_columns = []
for column in df:
    if df[column].dtype == object:
        categorical_columns.append(column)
le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])


#Training data
df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 100)


def conjugate_gradient(x0, Q,b, tole= 1e-6, max_iter= 10000):
  x = x0.copy()
  r = (Q @ x0) - b
  d = -r.copy()

  k = 0
  while np.linalg.norm(r) > tole and k + 1 < max_iter:
    numerator = np.dot(r, r.transpose())
    denominator = np.dot(np.dot(d, Q), d.transpose())
    alpha_k = numerator / denominator

    previous_r = r
    x = x + alpha_k * d
    r = r + alpha_k * np.dot(d, Q)
    beta = (r.transpose() @ r) / ((previous_r.transpose()) @ (previous_r))
    d = -r + beta * d
    k += 1
  return x

x_train = df_train.drop(columns= ['price'], axis= 1)
y_train = df_train['price']

A = np.c_[np.ones((len(x_train), 1)), x_train]

a = A.transpose() @ A
ab = A.transpose() @ y_train

x_star = conjugate_gradient(np.zeros(len(A[0])), a, ab)
##print(x_star)
def predict(X, x_star):
  X_with_intercept = np.c_[np.ones((len(X), 1)), X]
  return np.dot(X_with_intercept, x_star)

x_test = df_test.drop(columns= ['price'], axis= 1)
y_test = df_test['price']

y_predicted = predict(x_test, x_star)

r2 = r2_score(y_test, y_predicted)
print(f"R^2 Score = {r2}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predicted, alpha=0.5, color='blue', label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Perfect Fit Line")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()