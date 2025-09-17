import matplotlib.pyplot as plt
import numpy as np

data_path = "python-programming-PATRIK-HELLGREN/Labs/02_Programmering_inom_Python/Laboration_2/datapoints.txt"
clean_content = []

with open(data_path, "r") as f:
    next(f) # skip first row (headers)
    for data in f:
        split_content = data.split(",")
        row = [float(d.strip().replace("(", "").replace(")", "")) for d in split_content]
        clean_content.append(row)

x, y, z = zip(*clean_content)
x, y, z = np.array(x), np.array(y), np.array(z)

plt.scatter(x[z == 0], y[z == 0], color="green", alpha=0.6, edgecolors="black", label="Pichu", marker="o")
plt.scatter(x[z == 1], y[z == 1], color="yellow", alpha=0.6, edgecolors="black", label="Pikachu", marker="o")
    
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Pichu eller Pikachu?")
plt.grid(True)
plt.legend()
plt.show()

#content = pd.read_csv(data_path, delimiter=',')
# x = f["(width (cm)"]
# y = f[" height (cm)"]
# z = f[" label (0-pichu"]

# data_path = "python-programming-PATRIK-HELLGREN/Labs/02_Programmering_inom_Python/Laboration_2/testpoints.txt"
# df2 = pd.read_csv(data_path, skiprows=1, header=None, names=["testdata"]) # engine="python"

# pattern = r"\(\s*([+-]?(?:\d*\.\d+|\d+))\s*,\s*([+-]?(?:\d*\.\d+|\d+))\s*\)" # FEL!

# data_points = df2["testdata"].str.extract(pattern)
# data_points.columns = ["Width", "Height"]
# data_points = data_points.astype(float)
# print(data_points)

# plt.scatter(x[z == 0], y[z == 0], color="red", alpha=0.6, edgecolors="black", label="Pichu (test data)", marker="*")
# plt.scatter(x[z == 1], y[z == 1], color="blue", alpha=0.6, edgecolors="black", label="Pikachu (test data)", marker="*")

