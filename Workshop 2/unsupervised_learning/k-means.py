'''
Workshop #2 - AI Foundations

Welcome again! This is the code used in the second workshop. Here, you will see how
to classify models with no target information! For that, we use a technique called 
clusters, in which is done by the model K-means, an unsupervised learning model

As always, feel free to play around with this code :)

Use this to learn and remember to always have fun!
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Creating a DataFrame manually with customer data
# data = {
#     'Customer': ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Helen', 'Ian', 'Jack',
#                  'Kate', 'Leo', 'Mona', 'Nina', 'Oscar', 'Paul', 'Quinn', 'Rose', 'Steve', 'Tom'],
#     'Annual Spending ($)': [500, 700, 1200, 1500, 1800, 2200, 3000, 3400, 4000, 4500,
#                             5200, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000],
#     'Monthly Visits': [1, 2, 3, 3, 4, 5, 6, 6, 7, 8,
#                        9, 10, 10, 11, 12, 12, 13, 14, 15, 16]
# }

df = pd.DataFrame(data)  # Convert dictionary to pandas DataFrame

# Step 2: Applying K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = kmeans.fit_predict(df[['Annual Spending ($)', 'Monthly Visits']])

# Step 3: Visualizing the Clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Spending ($)'], df['Monthly Visits'], c=df['Cluster'], cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Spending ($)')
plt.ylabel('Monthly Visits')
plt.title('Customer Segmentation Using K-Means')
plt.legend()
plt.show()

# Display the DataFrame with clusters
print(df)