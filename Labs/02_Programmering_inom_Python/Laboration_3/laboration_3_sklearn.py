import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression # använda?
from sklearn.cluster import KMeans
from pathlib import Path

current_dir = Path(__file__).parent
data = pd.read_csv(current_dir/"unlabelled_data.csv", header=None, names=["x", "y", "Classification"])

def divide_into_clusters(data):
    data_points = data[["x", "y"]].values
    #print(data_points)
    kmeans = KMeans(n_clusters=2) # skapar en variabel för antal kluster och random seed.
    clusters = kmeans.fit_predict(data_points) # sätter 0 eller 1 beroende på klustertillhörighet
    #print(f"clusters: {clusters}")
    data["Classification"] = clusters # lägg till 0 eller 1 i en tredje kolumn ("Classification") på varje rad i data/CSV-filen.
    #print(data)
    return data, clusters # returnera datafilen inkl. den nya kolumnen samt 0:or och 1:or i clusters-variabeln.

def calculate_line(data): # utan scikit-learn (LogisticRegression)! Använda?
    
    # Split the xy values between cluster 1 and 2 depending on their label.
    cluster_1 = data[data["Classification"] == 0][["x", "y"]].values
    cluster_2 = data[data["Classification"] == 1][["x", "y"]].values
    #print(f"cluster_1: {cluster_1}")
    #print(f"cluster_2: {cluster_2}")

    # Calculate the mean of every cluster.
    mean_cluster_1 = cluster_1.mean(axis=0)
    mean_cluster_2 = cluster_2.mean(axis=0)
    #print(f"mean_cluster_1: {mean_cluster_1}")
    #print(f"mean_cluster_2: {mean_cluster_2}")

    # Calculate direction (the vector from midpoint 0 to midpoint 1).
    direction = mean_cluster_2 - mean_cluster_1 # (x2-x1), (y2-y1)
    #print(f"direction: {direction}")

    # Calculate the midpoint between the two cluster midpoints.
    midpoint = (mean_cluster_1 + mean_cluster_2) / 2 # (x1+x2)/2, (y1+y2)/2
    #print(f"midpoint: {midpoint}")

    # The dot product (skalärprodukten) between two vectors.
    b = -np.dot(direction, midpoint)
    #print(f"b: {b}")

    # Calculate k and m.
    k = -direction[0] / direction[1]
    #print(f"k: {-direction[0]} / {direction[1]} = {k}")
    m = -b / direction[1]
    #print(f"m: {-b} / {direction[1]} = {m}")
    x = np.linspace(-6, 6, 100)
    #print(f"k: {x}")
    y = k * x + m
    print(f"Line: y = {k:.2f}*x+{m:.2f}")

    return mean_cluster_1, mean_cluster_2, midpoint, y, x

def create_csv(data_with_classification):
    with open(current_dir/"labelled_data.csv", "w", newline="") as labelled_data:
        labelled_data.write(data_with_classification.to_csv(index=False, header=["x", "y", "Classification"]))

data_with_classification, clusters = divide_into_clusters(data)
#print(f"With Classification:\n {data_with_classification}")
mean_cluster_1, mean_cluster_2, midpoint, y, x = calculate_line(data_with_classification)
create_csv(data_with_classification)

# ta bort .iloc! Lös utan.
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