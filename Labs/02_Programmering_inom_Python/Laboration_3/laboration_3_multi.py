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

line_yx, labelled_data_yx = calculate_line_and_classify(-1.05, x, 0, data)
labelled_data = pd.DataFrame({"x": data["x"], "y": data["y"], "Classification: y(x)": labelled_data_yx})
create_csv(labelled_data)

line_fx, labelled_data_fx = calculate_line_and_classify(-0.489, x, 0, data)
line_gx, labelled_data_gx = calculate_line_and_classify(-2, x, 0.16, data)
line_hx, labelled_data_hx = calculate_line_and_classify(800, x, -120, data)

line_comparison = pd.DataFrame({"y(x)": labelled_data_yx, 
                                "f(x)": labelled_data_fx, 
                                "g(x)": labelled_data_gx, 
                                "h(x)": labelled_data_hx
                                })

#print(line_comparison.head().to_string(index=False))
#print(line_comparison.sum())

fig = plt.figure(figsize=(14,6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1,2,1], height_ratios=[1,1])
ax_left_top = fig.add_subplot(gs[0,0])
ax_left_bottom = fig.add_subplot(gs[1,0])
ax_center = fig.add_subplot(gs[:,1])
ax_right_top = fig.add_subplot(gs[0,2])
ax_right_bottom = fig.add_subplot(gs[1,2])

# Left top
ax_left_top.scatter(x, y, edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in line_comparison["y(x)"]])
ax_left_top.plot(x, line_yx, color="black", label="y(x) = -1.05x (my line)", linewidth=1)
ax_left_top.set_title("y(x) = -1.05x (my line)", fontsize=12)
ax_left_top.set_ylim(-6,6)
ax_left_top.set_xlim(-6,6)
ax_left_top.legend()

# Left bottom
ax_left_bottom.scatter(x, y, edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in line_comparison["f(x)"]])
ax_left_bottom.plot(x, line_fx, color="red", label="f(x) = -0.489x", linewidth=1)
ax_left_bottom.set_title("f(x) = -0.489x", fontsize=12)
ax_left_bottom.set_ylim(-6,6)
ax_left_bottom.set_xlim(-6,6)
ax_left_bottom.legend()

# Center
ax_center.scatter(x, y, edgecolors="black", alpha=0.4, color="red")
ax_center.plot(x, line_yx, color="black", label="y(x) = -1.05x (my line)", linewidth=1)
ax_center.plot(x, line_fx, color="red", label="f(x) = -0.489x", linewidth=1)
ax_center.plot(x, line_gx, color="green", label="g(x) = -2x + 0.16", linewidth=1)
ax_center.plot(x, line_hx, color="blue", label="h(x) = 800x - 120", linewidth=1)
ax_center.set_title("Laboration 3 - Linjär klassificering", fontweight="bold", fontsize=18)
ax_center.set_ylim(-6,6)
ax_center.set_xlim(-6,6)
ax_center.legend()

# Right top
ax_right_top.scatter(x, y, edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in line_comparison["g(x)"]])
ax_right_top.plot(x, line_gx, color="green", label="g(x) = -2x + 0.16", linewidth=1)
ax_right_top.set_title("g(x) = -2x + 0.16", fontsize=12)
ax_right_top.set_ylim(-6,6)
ax_right_top.set_xlim(-6,6)
ax_right_top.legend()

# Right bottom
ax_right_bottom.scatter(x, y, edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in line_comparison["h(x)"]])
ax_right_bottom.plot(x, line_hx, color="blue", label="h(x) = 800x - 120", linewidth=1)
ax_right_bottom.set_title("h(x) = 800x - 120", fontsize=12)
ax_right_bottom.set_ylim(-6,6)
ax_right_bottom.set_xlim(-6,6)
ax_right_bottom.legend()

plt.tight_layout()
plt.show()

# Eftersom h(x) är så brant (positivt k) så blir 0 -> 1 och 1 -> 0. Alla andra linjer har negativt k-värde!