import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
data = pd.read_csv(current_dir/"unlabelled_data.csv", header=None, names=["x", "y", "Classification"])
x = np.array(data["x"])
y = np.array(data["y"])

def calculate_line_and_classify(k, x, m, data=None):

    line = k * x + m # om x = 0 -> line = m -> linjen skär y på m när x = 0.

    if data is not None:
        data["Classification"] = (data["y"] > line).astype(int)
        
        return line, data

    return line

def create_csv(data):

    try:
        with open(current_dir/"labelled_data.csv", "w", newline="") as labelled_data:
            labelled_data.write(data.to_csv(index=False, header=["x", "y", "Classification"]))
    except OSError as err:
        print(f"Something went wrong while writing to file: {err}.")

line_ph, labelled_data_ph = calculate_line_and_classify(-1.05, x, 0, data) # bara höfta -1.05??
line_fx, labelled_data_fx = calculate_line_and_classify(-0.489, x, 0, data)
line_gx, labelled_data_gx = calculate_line_and_classify(-2, x, 0.16, data)
line_hx, labelled_data_hx = calculate_line_and_classify(800, x, -120, data)
# -> Jämföra labelled data ph, fx, gx och hx. Hur? Skapa flera figures -> fig1.scatter -> byt ut labelled_data_* för färgsättning?
create_csv(labelled_data_ph)

plt.scatter(data["x"], data["y"], edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in labelled_data_ph["Classification"]])
plt.plot(x, line_ph, color="black", label="My line = -1.05x", linewidth=1)
plt.plot(x, line_fx, color="red", label="f(x) = -0.489x", linewidth=1)
plt.plot(x, line_gx, color="green", label="g(x) = -2x + 0.16", linewidth=1)
plt.plot(x, line_hx, color="blue", label="h(x) = 800x - 120", linewidth=1)
plt.title("Laboration 3", fontweight="bold")
plt.ylim(-6,6)
plt.xlim(-6,6)
plt.legend()
plt.tight_layout()
plt.show()