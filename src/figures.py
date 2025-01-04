import numpy as np
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt

# Generate some data
np.random.seed(0)
data = np.random.normal(0, 1, 100)
data = np.append(data, [10])  # Adding global outliers

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data, 'bo', label='Data points')
# plt.axhline(y=np.mean(data), color='r', linestyle='-', label='Mean')
# plt.axhline(y=np.mean(data) + 3*np.std(data), color='g', linestyle='--', label='Mean + 3*Std Dev')
# plt.axhline(y=np.mean(data) - 3*np.std(data), color='g', linestyle='--', label='Mean - 3*Std Dev')

# Highlight the outliers
outliers = data[(data > np.mean(data) + 3*np.std(data)) | (data < np.mean(data) - 3*np.std(data))]
plt.plot(np.where((data > np.mean(data) + 3*np.std(data)) | (data < np.mean(data) - 3*np.std(data)))[0], outliers, 'ro', label='Global Outliers')

# plt.title('Global Outliers Detection')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()


# Generate some data
np.random.seed(0)
data = np.random.normal(0, 1, 100)
data = np.append(data, [8, 9, 10])  # Adding global outliers

# Reshape data for LOF
data = data.reshape(-1, 1)

# Fit the model
lof = LocalOutlierFactor(n_neighbors=20)
outlier_labels = lof.fit_predict(data)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data, 'bo', label='Data points')

# Highlight the local outliers
local_outliers = data[outlier_labels == -1]
plt.plot(np.where(outlier_labels == -1)[0], local_outliers, 'ro', label='Local Outliers')

# plt.title('Local Outliers Detection using LOF')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Generate some data with collective outliers relative to other clusters
np.random.seed(0)
data_cluster_1 = np.random.normal(0, 1, 70)
data_cluster_2 = np.random.normal(5, 0.5, 100)
collective_outliers = np.random.normal(10, 0.1, 10)  # Adding collective outliers
data = np.concatenate((data_cluster_1, data_cluster_2, collective_outliers))

# Create x values
x_values = np.arange(len(data))

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(x_values, data, 'bo', label='Data points')
# plt.axhline(y=np.mean(data), color='r', linestyle='-', label='Mean')
# plt.axhline(y=np.mean(data) + 3*np.std(data), color='g', linestyle='--', label='Mean + 3*Std Dev')
# plt.axhline(y=np.mean(data) - 3*np.std(data), color='g', linestyle='--', label='Mean - 3*Std Dev')

# Highlight the collective outliers
collective_outliers_indices = np.arange(len(data) - len(collective_outliers), len(data))
plt.plot(collective_outliers_indices, collective_outliers, 'ro', label='Collective Outliers')

# plt.title('Collective Outliers Detection Relative to Clusters')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()