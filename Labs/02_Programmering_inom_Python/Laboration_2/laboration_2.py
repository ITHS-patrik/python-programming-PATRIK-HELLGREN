import matplotlib.pyplot as plt
import numpy as np

data_path_training = "python-programming-PATRIK-HELLGREN/Labs/02_Programmering_inom_Python/Laboration_2/datapoints.txt"
data_path_test = "python-programming-PATRIK-HELLGREN/Labs/02_Programmering_inom_Python/Laboration_2/testpoints.txt"
clean_content_tr = []
clean_content_te = []

# TRAINING DATA (tr)
with open(data_path_training, "r") as tr:
    next(tr) # skip first row (headers)
    for tr_data in tr:
        split_content_tr = tr_data.split(",")
        tr_row = [float(d.strip().replace("(", "").replace(")", "")) for d in split_content_tr]
        clean_content_tr.append(tr_row)

tr_x, tr_y, tr_z = zip(*clean_content_tr)
tr_x, tr_y, tr_z = np.array(tr_x), np.array(tr_y), np.array(tr_z)

plt.scatter(tr_x[tr_z == 0], tr_y[tr_z == 0], color="green", alpha=0.6, edgecolors="black", label="Pichu", marker="o")
plt.scatter(tr_x[tr_z == 1], tr_y[tr_z == 1], color="yellow", alpha=0.6, edgecolors="black", label="Pikachu", marker="o")

# TEST DATA (te)
# ta bort alla prints när labben är klar
with open(data_path_test, "r") as te:
    next(te) # skip first row (header)
    for te_data in te:
        te_data = te_data.split(" ", 1)[1]
        split_content_te = [float(d.strip().replace("(", "").replace(")", "")) for d in te_data.split(",")]
        clean_content_te.append(split_content_te)

print(clean_content_te)

te_x, te_y = zip(*clean_content_te)
print(f"te_x, no np array: {te_x}")
print(f"te_y, no np array: {te_y}")
te_x, te_y = np.array(te_x), np.array(te_y)
print(f"te_x, np array: {te_x}")
print(f"te_y, np array: {te_y}")

# NEAREST NEIGHBOUR
# ta bort alla prints när labben är klar
tr_points = np.column_stack((tr_x, tr_y))
print(f"tr_points:\n{tr_points}")
tr_labels = tr_z
print(f"tr_labels:\n{tr_labels}")
te_points = np.column_stack((te_x, te_y))
print(f"te_points:\n{te_points}")

predicitons = []

for te_point in te_points:
    diff_x_y = tr_points[:, np.newaxis, :] - te_points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff_x_y ** 2, axis=2))
    nearest_id = np.argmin(distances, axis=0)
    prediction_label = tr_labels[nearest_id]
    predicitons.append(prediction_label)

print(f"distances:\n{distances}")
print(f"nearest_id:\n{nearest_id}")
print(f"prediction_label:\n{prediction_label}")
print(f"predictions, no np array:\n{predicitons}")
predicitons = np.array(predicitons)
print(f"predictions, np array:\n{predicitons}")

# PLOT

#plt.scatter(te_x[te_z == 0], te_y[te_z == 0], color="green", alpha=0.6, edgecolors="black", label="Pichu", marker="*")
#plt.scatter(te_x[te_z == 1], te_y[te_z == 1], color="yellow", alpha=0.6, edgecolors="black", label="Pikachu", marker="*")

# plt.xlabel("Width")
# plt.ylabel("Height")
# plt.title("Pichu eller Pikachu?")
# plt.grid(True)
# plt.legend()
# plt.show()