import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
data = pd.read_csv(current_dir/"unlabelled_data.csv", header=None, names=["x", "y", "Classification"])
x = np.array(data["x"])
y = np.array(data["y"])

def calculate_midpoints(x, y, data):
    
    midpoint = (np.median(x), np.median(y))
    print(f"midpoint: {midpoint}")

    cluster_1 = data.loc[data["Classification"] == 0, ["x", "y"]].values
    #print(f"cluster_1: {cluster_1}")
    midpoint_cluster_1 = np.mean(cluster_1, axis=0)
    print(f"midpoint_cluster_1: {midpoint_cluster_1}")

    cluster_2 = data.loc[data["Classification"] == 1, ["x", "y"]].values
    midpoint_cluster_2 = np.mean(cluster_2, axis=0)
    print(f"midpoint_cluster_2: {midpoint_cluster_2}")
    
    return midpoint_cluster_1, midpoint_cluster_2, midpoint

def calculate_line(x, y, k, m):

    line = k * x + m # om x = 0 -> line = m -> linjen skär y på m när x = 0.

    #return  0 if line > y else 1
    return line

def create_csv(data):

    try:
        with open(current_dir/"labelled_data.csv", "w", newline="") as classified_data:
            classified_data.write(data.to_csv(index=False, header=["x", "y", "Classification"]))
    except FileNotFoundError as err: # ANVÄNDA NÅGOT ANNAT ÄN FILENOTFOUNDERROR?
        print(f"Something went wrong while writing to file {classified_data}: {err}.")

def classify_data_point(data, k):

    data["Classification"] = np.where(data["y"] > k * data["x"], 1, 0)

    return data

classified_data = classify_data_point(data, -1.05)
midpoint_cluster_1, midpoint_cluster_2, midpoint = calculate_midpoints(x, y, classified_data)
create_csv(classified_data)

# lines:
line_ph = calculate_line(x, y, -1.05, 0)
line_fx = calculate_line(x, y, -0.489, 0)
line_gx = calculate_line(x, y, -2, 0.16)
line_hx = calculate_line(x, y, 800, -120)

plt.scatter(data["x"], data["y"], edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in classified_data["Classification"]])
plt.scatter(*midpoint_cluster_1, color="cyan", edgecolors="black", label="Midpoint in cluster 1")
plt.scatter(*midpoint_cluster_2, color="yellow", edgecolors="black", label="Midpoint in cluster 2")
plt.scatter(*midpoint, color="red", edgecolors="black", marker="*", s=100, label="Midpoint between clusters")

plt.plot(x, line_ph, color="black", label="My line")
plt.plot(x, line_fx, color="red", label="line_fx")
plt.plot(x, line_gx, color="green", label="line_gx")
plt.plot(x, line_hx, color="blue", label="line_hx")
plt.title("Laboration 3", fontweight="bold")
plt.ylim(-6,6)
plt.xlim(-6,6)
plt.legend()
plt.tight_layout()
plt.show()