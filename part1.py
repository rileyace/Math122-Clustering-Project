from sklearn.cluster import KMeans
from tensorflow.keras.datasets import boston_housing
from skimage.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

# Scales the data before fitting to model --> we can discuss if we need this
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

kmeans = KMeans(n_clusters=3)
kmeans.fit(x_train_scaled)
clusters = kmeans.predict(x_train_scaled)

plt.figure()
plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=clusters)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KMeans Clusters (First Two Features)")

plt.show()

#-------------------- part iii -----------------------------
