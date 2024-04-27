import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

clearance_data = pd.read_csv("CollisionData.csv")
X = clearance_data[[c for c in clearance_data.columns if "Theta" in c]].copy()
for col_idx, col in enumerate(X.columns):
    clearance_data[f"Embed{2*col_idx}"] = np.cos(clearance_data[col])
    clearance_data[f"Embed{2*col_idx+1}"] = np.sin(clearance_data[col])
    clearance_data[f"BP{2*col_idx}"] = np.cos(clearance_data[col])
    clearance_data[f"BP{2*col_idx+1}"] = np.sin(clearance_data[col])
X_embed = clearance_data[[c for c in clearance_data.columns if "Embed" in c]].copy()
Y = clearance_data[[c for c in clearance_data.columns if "to" in c]].copy()

normal = []
embed = []
for _ in range(20):

    model = Sequential()
    model.add(Dense(5, input_dim=X_embed.shape[1], activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(12,  activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    history = model.fit(X_embed, Y, epochs=5, verbose=1)
    embed.append(history.history["mse"])
    plt.plot(history.history["mse"], label="embed", c="r", linewidth=1, alpha=0.5)

    model = Sequential()
    model.add(Dense(5, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(12,  activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    history = model.fit(X, Y, epochs=5, verbose=1)
    normal.append(history.history["mse"])
    plt.plot(history.history["mse"], label="normal", c="b", linewidth=1, alpha=0.5)


plt.show()