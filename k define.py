import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Sample data (replace this with your dataset)
file_path = "C:/Users/homa.behmardi/Downloads/eachbandbehaviour.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")

# Assuming 'SITE' is the categorical column in your DataFrame
# Perform one-hot encoding
encoded_columns = pd.get_dummies(data['sector'], prefix='sector')
encoded_data = pd.concat([data[['Payload', 'Throughput']], encoded_columns], axis=1)

# Determine the optimal number of clusters (k) using the elbow method
inertia_values = []
possible_k_values = range(1, 11)  # Trying k values from 1 to 10

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(encoded_data)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(possible_k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose the best k based on the elbow curve (you need to visually inspect it)
best_k = 10  # Change this value based on your observation

# Initialize KMeans object with the best k
kmeans = KMeans(n_clusters=best_k)

# Fit the model to the data
kmeans.fit(encoded_data)

# Get cluster assignments and cluster centers
cluster_labels = kmeans.labels_

# Add the cluster labels to the original DataFrame
encoded_data['Cluster'] = cluster_labels

# Visualize the results (for 2D data)
x = encoded_data['Payload']
y = encoded_data['Throughput']
cluster_labels = encoded_data['Cluster']

# Create a scatter plot
plt.scatter(x, y, c=cluster_labels, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='black', s=100)  # Cluster centers
plt.xlabel('Payload')
plt.ylabel('Throughput')
plt.title('K-Means Clustering')
plt.show()

# Assuming 'encoded_data' is your DataFrame with the clustering results
cluster_1_data = encoded_data[encoded_data['Cluster'] == 1]

# # Display the data points in cluster 1
# print(cluster_1_data)

# # Create a dictionary to store site-to-cluster mapping
# site_to_cluster = {}

# # Populate the site-to-cluster dictionary
# for index, row in encoded_data.iterrows():
#     site = row['site']
#     cluster = row['Cluster']
#     if site not in site_to_cluster:
#         site_to_cluster[site] = cluster

# # Display complete rows for each site and its corresponding cluster
# for site, cluster in site_to_cluster.items():
#     site_data = encoded_data[encoded_data['site'] == site]
#     print(f"Site: {site}, Cluster: {cluster}")
#     print(site_data)
#     print("====================")
