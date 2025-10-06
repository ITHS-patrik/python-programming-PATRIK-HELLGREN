import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

current_dir = Path(__file__).parent

def read_data():
    try:
        data = pd.read_csv(current_dir/"unlabelled_data.csv", header=None, names=["x", "y"])
    except OSError as err:
        print(f"Something went wrong while reading the data file:\n{err}.\nExiting program.")
        sys.exit(1)

    x = np.array(data["x"])
    y = np.array(data["y"])

    return data, x, y

def calculate_line_and_classify(k, x, m, data):
    df = data.copy() # creates a copy to not overwrite the original DataFrame later on during line comparison.

    line = k * x + m
    df["Classification"] = (df["y"] > line).astype(int)

    return line, df["Classification"]

def create_csv(data, labelled_data_yx):
    
    labelled_data = pd.DataFrame({"x": data["x"], "y": data["y"], "Classification: y(x)": labelled_data_yx})
    labelled_data.to_csv(current_dir/"labelled_data.csv", index=False)

    return labelled_data

def plot_separate_line(x, y, line_labels, column, line, color, title, ax_pos=plt):

    ax_pos.scatter(x, y, edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in line_labels[column]])
    ax_pos.scatter([], [], edgecolors="black", alpha=0.6, color="green", label=rf"Classification 0: $\bf{len(line_labels[column]) - line_labels[column].sum()}$ data points")
    ax_pos.scatter([], [], edgecolors="black", alpha=0.6, color="blue", label=rf"Classification 1: $\bf{line_labels[column].sum()}$ data points")
    ax_pos.plot(x, line, color=color, linewidth=1)
    ax_pos.legend(fontsize=8, loc="upper left")

    if ax_pos == plt:    
        ax_pos.title(title, fontsize=10, fontweight="bold")
        ax_pos.xlim(-6, 6)
        ax_pos.ylim(-6, 6)
        ax_pos.show()
    else:
        ax_pos.set_title(title, fontsize=10)
        ax_pos.set_xlim(-6, 6)
        ax_pos.set_ylim(-6, 6)

data, x, y = read_data()
line_yx, labelled_data_yx = calculate_line_and_classify(-1.05, x, 0.3, data)
labelled_data = create_csv(data, labelled_data_yx)

if __name__ == "__main__":
    
    # Plot my line along with labelled data points
    plot_separate_line(x, y, labelled_data, "Classification: y(x)", line_yx, "black", "y(x) = -1.05x + 0.3")
