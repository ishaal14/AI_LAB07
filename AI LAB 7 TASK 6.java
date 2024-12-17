import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#Sample dataset of house features (square footage, number of bedrooms, age of the house, price)
data = {
    'Square Feet': [2000, 2500, 1800, 2200, 1600, 2100, 2400, 2800, 3000, 1900],
    'Bedrooms': [3, 4, 3, 3, 2, 3, 4, 5, 5, 3],
    'Age_of_House': [10, 15, 8, 20, 5, 15, 30, 25, 35, 10],
    'Price': [400000, 500000, 350000, 450000, 300000, 430000, 550000, 600000, 650000, 380000]
}
#Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
#Features (excluding the 'Price' for clustering, because we are predicting price from clusters)
X = df.drop(columns=['Price'])
#Normalize the features (important for K-means to perform well)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Apply K-means clustering to the data
#We will try to find 3 clusters for this example (K-3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
#Add cluster centers (centroids)
df_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
df_centroids['Cluster'] = df_centroids.index
#Output the clusters and their corresponding features
print("House Clusters:")
print(df)
