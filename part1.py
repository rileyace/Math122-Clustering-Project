from sklearn.cluster import KMeans
from tensorflow.keras.datasets import boston_housing
from skimage.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# ------------------- Code for part i ----------------------
model = Ridge(alpha=.5)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")


#-------------------- part ii ------------------------------

kmeans = KMeans(n_clusters=3)
kmeans.fit(x_train)
clusters = kmeans.predict(x_train)


plt.figure()
plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KMeans Clusters (First Two Features)")

plt.show()

#-------------------- part iii -----------------------------
test_clusters = kmeans.predict(x_test)

models = []

y_test_pred = np.zeros_like(y_test)

for cluster_id in range(3):

    train_mask = clusters == cluster_id
    x_cluster = x_train[train_mask]
    y_cluster = y_train[train_mask]

    ridge_model = Ridge(alpha=0.5)
    ridge_model.fit(x_cluster, y_cluster)
    models.append(ridge_model)

    test_mask = test_clusters == cluster_id

    x_test_cluster = x_test[test_mask]

    y_test_pred[test_mask] = ridge_model.predict(x_test_cluster)

clustered_mse = mean_squared_error(y_test, y_test_pred)
print(f"Clustered Ridge Mean Squared Error: {clustered_mse:.4f}")