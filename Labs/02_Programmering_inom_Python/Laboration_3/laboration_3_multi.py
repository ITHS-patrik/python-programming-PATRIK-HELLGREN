import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
from pathlib import Path

current_dir = Path(__file__).parent
data = pd.read_csv(current_dir/"unlabelled_data.csv", header=None, names=["x", "y"])
x = np.array(data["x"])
y = np.array(data["y"])

def calculate_line_and_classify(k, x, m, data):
    df = data.copy() # creates a copy to not overwrite the original DataFrame later on during line comparision.

    line = k * x + m
    df["Classification"] = (df["y"] > line).astype(int)

    return line, df["Classification"]

def create_csv(data):

    try:
        data.to_csv(current_dir/"labelled_data.csv", index=False)
    except OSError as err:
        print(f"Something went wrong while writing to file: {err}.\nExiting program.")
        sys.exit(1)

def calculate_accuracy(line_comparison):
    
    lines = ["f(x)", "g(x)", "h(x)"]
    accuracies = [(line_comparison[m] == line_comparison["y(x)"]).mean() for m in lines]
    accuracy_data = pd.DataFrame({"Line": lines, "Accuracy": accuracies})
    plt.bar(accuracy_data["Line"], accuracy_data["Accuracy"]*100, color=["green", "blue", "orange"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.suptitle("Accuracy per line", fontweight="bold")

    for index, value in enumerate(accuracy_data["Accuracy"]):
        plt.text(index, value*100 + 1, f"{value*100:.2f}%", ha="center")

def plot_separate_lines(ax_pos, x, y, line_comparison, column, line, color, title):

    ax_pos.scatter(x, y, edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in line_comparison[column]])
    ax_pos.scatter([], [], edgecolors="black", alpha=0.6, color="green", label=rf"Classification 0: $\bf{len(line_comparison[column]) - line_comparison[column].sum()}$")
    ax_pos.scatter([], [], edgecolors="black", alpha=0.6, color="blue", label=rf"Classification 1: $\bf{line_comparison[column].sum()}$")
    ax_pos.plot(x, line, color=color, linewidth=1)

    ax_pos.set_title(title, fontsize=12)
    ax_pos.set_xlim(-6, 6)
    ax_pos.set_ylim(-6, 6)
    ax_pos.legend(fontsize=8, loc="upper left")

line_yx, labelled_data_yx = calculate_line_and_classify(-1.05, x, 0, data)
labelled_data = pd.DataFrame({"x": data["x"], "y": data["y"], "Classification: y(x)": labelled_data_yx})
create_csv(labelled_data)

line_fx, labelled_data_fx = calculate_line_and_classify(-0.489, x, 0, data)
line_gx, labelled_data_gx = calculate_line_and_classify(-2, x, 0.16, data)
line_hx, labelled_data_hx = calculate_line_and_classify(800, x, -120, data)

# Data used for plotting
line_comparison = pd.DataFrame({"y(x)": labelled_data_yx, 
                                "f(x)": labelled_data_fx, 
                                "g(x)": labelled_data_gx, 
                                "h(x)": labelled_data_hx
                                })
yfg_lines_equal = line_comparison.iloc[:, :3].nunique(axis=1) == 1
all_lines_equal = line_comparison.nunique(axis=1) == 1

fig = plt.figure(figsize=(14,6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1,2,1], height_ratios=[1,1])
ax_left_top =       fig.add_subplot(gs[0,0])
ax_left_bottom =    fig.add_subplot(gs[1,0])
ax_center =         fig.add_subplot(gs[:,1])
ax_right_top =      fig.add_subplot(gs[0,2])
ax_right_bottom =   fig.add_subplot(gs[1,2])

# Plot side plots
plot_separate_lines(ax_left_top, x, y, line_comparison, "y(x)", line_yx, "black", "y(x) = -1.05x")
plot_separate_lines(ax_left_bottom, x, y, line_comparison, "f(x)", line_fx, "red", "f(x) = -0.489x")
plot_separate_lines(ax_right_top, x, y, line_comparison, "g(x)", line_gx, "green", "g(x) = -2x + 0.16")
plot_separate_lines(ax_right_bottom, x, y, line_comparison, "h(x)", line_gx, "blue", "h(x) = 800x - 120")

# Plot center/main (all lines)
ax_center.scatter(x, y, edgecolors="black", alpha=0.6, color=["red" if same else ("orange" if yfg_same else "cyan") for same, yfg_same in zip(all_lines_equal, yfg_lines_equal)])
ax_center.scatter([], [], edgecolors="black", alpha=0.6, color="red", label="Always same classification for all lines")
ax_center.scatter([], [], edgecolors="black", alpha=0.6, color="cyan", label="Not always same classification for lines k < 0")
ax_center.plot(x, line_yx, color="black", label="y(x) = -1.05x (my line)", linewidth=1)
ax_center.plot(x, line_fx, color="red", label="f(x) = -0.489x", linewidth=1)
ax_center.plot(x, line_gx, color="green", label="g(x) = -2x + 0.16", linewidth=1)
ax_center.plot(x, line_hx, color="blue", label="h(x) = 800x - 120", linewidth=1)
ax_center.set_title("Laboration 3 - Linjär klassificering", fontweight="bold", fontsize=18)
ax_center.set_ylim(-6,6)
ax_center.set_xlim(-6,6)
ax_center.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()

calculate_accuracy(line_comparison)
plt.show()
