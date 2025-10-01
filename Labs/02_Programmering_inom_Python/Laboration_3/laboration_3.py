import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
data = pd.read_csv(current_dir/"unlabelled_data.csv", header=None, names=["x", "y"])
x = np.array(data["x"])
y = np.array(data["y"])

def calculate_line_and_classify(k, x, m, data):
    df = data.copy() # creates a copy to not overwrite the original DataFrame later on during line comparision.

    line = k * x + m
    df["Classification"] = (df["y"] > line).astype(int)

    return line, df

def create_csv(data):

    try:
        data.to_csv(current_dir/"labelled_data.csv", index=False)
    except OSError as err:
        print(f"Something went wrong while writing to file: {err}.")

line_ph, labelled_data_ph = calculate_line_and_classify(-1.05, x, 0, data)
line_fx, labelled_data_fx = calculate_line_and_classify(-0.489, x, 0, data)
line_gx, labelled_data_gx = calculate_line_and_classify(-2, x, 0.16, data)
line_hx, labelled_data_hx = calculate_line_and_classify(800, x, -120, data)

# -> Jämföra labelled data ph, fx, gx och hx. Hur? Skapa flera figures -> fig1.scatter -> byt ut labelled_data_* för färgsättning?
# -> Jämföra antalet punkter under/över linjen mellan de olika linjerna? Typ count += 1 per linje?

create_csv(labelled_data_ph)

plt.scatter(x, y, edgecolors="black", alpha=0.6, color=["green" if i == 0 else "blue" for i in labelled_data_ph["Classification"]])
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