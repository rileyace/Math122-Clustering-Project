from tensorflow.keras.datasets import boston_housing
from skimage.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import numpy as np


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# ------------------- Code for part a ----------------------
model = Ridge(alpha=.5)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

