'''
Workshop #2 - AI Foundations

Welcome again! This is the code used in the second workshop. Here, you will see how
to classify models with no target information! For that, we use a technique called 
clusters, in which is done by the model K-means, an unsupervised learning model

As always, feel free to play around with this code :)

Use this to learn and remember to always have fun!
'''
import pandas as pd #Library to Work with Data
import matplotlib.pyplot as plt #Library to plot our Results
from sklearn.cluster import KMeans #Library that contains our ML model

#1) IMPORTING THE DATA ------------------------------------------------------------------
data = {
    'Customer': ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Helen', 'Ian', 'Jack',
                 'Kate', 'Leo', 'Mona', 'Nina', 'Oscar', 'Paul', 'Quinn', 'Rose', 'Steve', 'Tom'],
    'Monthly Visits': [1, 2, 3, 3, 4, 5, 9, 13, 11, 12,
                       14, 28, 21, 26, 20, 24, 28, 29, 35, 37],
    'Annual Spending ($)': [500, 700, 1200, 1500, 1850, 1250, 3000, 3400, 4000, 4500,
                            5200, 6000, 6500, 7000, 8200, 7700, 7900, 8300, 9500, 8700]
}

df = pd.DataFrame(data)  # Convert dictionary to pandas DataFrame

#2) TRAINING DATA -----------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)

#3) PREDICTING DATA ---------------------------------------------------------------------
cluster = kmeans.fit_predict(df[['Monthly Visits', 'Annual Spending ($)']])

#4) PLOTTING RESULTS --------------------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(df['Monthly Visits'], df['Annual Spending ($)']) #Plotting Data without Clusters (make sure to uncomment this)

plt.scatter(df['Monthly Visits'], df['Annual Spending ($)'], c= cluster, cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')

#Plotting Details
plt.xlabel('Monthly Visits')
plt.ylabel('Annual Spending ($)')
plt.title('Customer Segmentation Using K-Means')
plt.legend()
plt.show()

# Display the DataFrame with clusters
print(df)