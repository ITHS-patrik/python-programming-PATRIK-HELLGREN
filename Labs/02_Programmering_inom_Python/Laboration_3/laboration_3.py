import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from pathlib import Path

current_dir = Path(__file__).parent
data = pd.read_csv(current_dir/"unlabelled_data.csv", header=None, names=["x", "y", "Classification"])

def divide_into_clusters(data):
    data_points = data[["x", "y"]].values
    #print(data_points)
    kmeans = KMeans(n_clusters=2, random_state=1337)
    clusters = kmeans.fit_predict(data_points)
    #print(f"clusters: {clusters}")
    data["Classification"] = clusters
    #print(data)
    return data, clusters

def calculate_line(data): # utan scikit-learn (LogisticRegression)! Använda?
    
    # Split the xy values between cluster 1 and 2 depending on their label.
    cluster_1 = data[data["Classification"] == 0][["x", "y"]].values
    cluster_2 = data[data["Classification"] == 1][["x", "y"]].values

    # Calculate the mean of every cluster.
    mean_cluster_1 = cluster_1.mean(axis=0)
    mean_cluster_2 = cluster_2.mean(axis=0)

    # Calculate direction (the vector from midpoint 0 to midpoint 1).
    direction = mean_cluster_2 - mean_cluster_1 # (x2-x1), (y2-y1)

    # Calculate the midpoint between the two cluster midpoints.
    midpoint = (mean_cluster_1 + mean_cluster_2) / 2 # (x1+x2)/2, (y1+y2)/2

    # The dot product (skalärprodukten) between two vectors.
    b = -np.dot(direction, midpoint)

    # Calculate k and m.
    k = -direction[0] / direction[1]
    m = -b / direction[1]
    x = np.linspace(-6, 6, 100)
    y = k * x + m
    print(f"Line: y = {k:.2f}*x+{m:.2f}")

    return mean_cluster_1, mean_cluster_2, midpoint, y, x

data_with_classification, clusters = divide_into_clusters(data)
#print(f"With Classification:\n {data_with_classification}")
mean_cluster_1, mean_cluster_2, midpoint, y, x = calculate_line(data_with_classification)

plt.scatter(data.iloc[:, 0], data.iloc[:, 1], edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in clusters])
plt.scatter(*mean_cluster_1, color="cyan", edgecolors="black", label="Midpoint in cluster 1")
plt.scatter(*mean_cluster_2, color="yellow", edgecolors="black", label="Midpoint in cluster 2")
plt.scatter(*midpoint, color="red", marker="*", s=100, label="Midpoint between clusters")

plt.plot(x, y, color="black", linestyle="--", label="Line")
plt.title("Laboration 3", fontweight="bold")
plt.ylim(-6,6)
plt.xlim(-6,6)
plt.legend()
plt.tight_layout()
plt.show()