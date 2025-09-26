import matplotlib.pyplot as plt
import numpy as np

data_path_training = "python-programming-PATRIK-HELLGREN/Labs/02_Programmering_inom_Python/Laboration_2/datapoints.txt"
data_path_test = "python-programming-PATRIK-HELLGREN/Labs/02_Programmering_inom_Python/Laboration_2/testpoints.txt"
clean_content_tr = []
clean_content_te = []

# TRAINING DATA (tr)
with open(data_path_training, "r") as tr:
    next(tr)
    for tr_data in tr:
        split_content_tr = tr_data.split(",")
        tr_row = [float(d.strip().replace("(", "").replace(")", "")) for d in split_content_tr]
        clean_content_tr.append(tr_row)

tr_x, tr_y, tr_z = zip(*clean_content_tr)
tr_x, tr_y, tr_z = np.array(tr_x), np.array(tr_y), np.array(tr_z)

# TEST DATA (te)
with open(data_path_test, "r") as te:
    next(te)
    for te_data in te:
        te_data = te_data.split(" ", 1)[1]
        split_content_te = [float(d.strip().replace("(", "").replace(")", "")) for d in te_data.split(",")]
        clean_content_te.append(split_content_te)

te_x, te_y = zip(*clean_content_te)
te_x, te_y = np.array(te_x), np.array(te_y)

# NEAREST NEIGHBOUR
tr_points = np.column_stack((tr_x, tr_y))
tr_labels = tr_z
te_points = np.column_stack((te_x, te_y))

predictions = []

for te_point in te_points:
    diff_x_y = tr_points[:, np.newaxis, :] - te_point
    distances = np.sqrt(np.sum(diff_x_y ** 2, axis=2))
    nearest_id = np.argmin(distances, axis=0)
    prediction_label = tr_labels[nearest_id]
    predictions.append(prediction_label)

predictions = np.array(predictions).ravel()

# TEST DATA CLASSIFICATION (1-NN)
for i in range(len(te_x)):
    pokemon = "Pichu" if predictions[i] == 0 else "Pikachu"
    print(f"Sample with (width, height): ({te_x[i]}, {te_y[i]}) classified as {pokemon}")

# PLOT
## Training data
plt.scatter(tr_x[tr_z == 0], tr_y[tr_z == 0], color="green", alpha=0.6, edgecolors="black", label="Pichu", marker="o")
plt.scatter(tr_x[tr_z == 1], tr_y[tr_z == 1], color="yellow", alpha=0.6, edgecolors="black", label="Pikachu", marker="o")

## Test data
plt.scatter(te_x[predictions == 0], te_y[predictions == 0], color="green", edgecolors="black", label="Pichu (test)", marker="^")
plt.scatter(te_x[predictions == 1], te_y[predictions == 1], color="yellow", edgecolors="black", label="Pikachu (test)", marker="^")

plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Pichu eller Pikachu?")
plt.grid(True)

# 1. USER INPUT 1-NN & EXCEPTION HANDLING
while True:
    try:
        raw_input_x = input("Please input the x value (width) for your test data point: ")
        raw_input_y = input("Please input the y value (height) for your test data point: ")

        refined_input_x = float(raw_input_x)
        refined_input_y = float(raw_input_y)
        
        if refined_input_x <= 0 or refined_input_y <= 0:
            raise ValueError("NegativeValue")
        
    except ValueError as e:
        if str(e) == "NegativeValue":
            print(f"Null or negative values are not accepted.\nYour input was: ({raw_input_x}, {raw_input_y}).\nTry again.")
        else:
            print(f'Your input "({raw_input_x}, {raw_input_y})" contains invalid characters! Only use floats > 0.\nTry again.')

    else:
        input_data_point = np.array([refined_input_x, refined_input_y])
        input_diff = tr_points - input_data_point
        distances = np.sqrt(np.sum(input_diff ** 2, axis=1))
        nearest_id = np.argmin(distances)
        prediction_label = tr_labels[nearest_id]
        pokemon = "Pichu" if prediction_label == 0 else "Pikachu"

        print(f"Thank you!")
        print(f"Your input ({refined_input_x}, {refined_input_y}) classifies as {pokemon} with 1-NN.")

        ## Plot input_data_point 1-NN
        if pokemon == "Pichu":
            plt.scatter(refined_input_x, refined_input_y, color="red", edgecolors="black", label="Pichu (user_input 1-NN)", marker="*", s=175)
        else:
            plt.scatter(refined_input_x, refined_input_y, color="red", edgecolors="black", label="Pikachu (user_input 1-NN)", marker="*", s=175)
        break

# 2. USER INPUT 10-NN
refined_input_x_10NN = refined_input_x
refined_input_y_10NN = refined_input_y
input_data_point_10NN = np.array([refined_input_x_10NN, refined_input_y_10NN])

ten = 10
nearest_ids = np.argsort(distances)[:ten]
nearest_ids_labels = tr_labels[nearest_ids]

label_count = np.bincount(nearest_ids_labels.astype(int))
prediction_label = np.argmax(label_count)

pokemon = "Pichu" if prediction_label == 0 else "Pikachu"

print(f"Your input ({refined_input_x_10NN}, {refined_input_y_10NN}) classifies as {pokemon} with 10-NN.")

## Plot input_data_point 1-NN
if pokemon == "Pichu":
    plt.scatter(refined_input_x, refined_input_y, color="red", edgecolors="black", label="Pichu (user_input 10-NN)", marker="*", s=175)
else:
    plt.scatter(refined_input_x, refined_input_y, color="red", edgecolors="black", label="Pikachu (user_input 10-NN)", marker="*", s=175)

## Reflection
reflection_x = 21.8
reflection_y = 33.5
print(f"Reflection: when the user types in e.g. ({reflection_x}, {reflection_y}) the model classifies the pokemon as Pichu with 1-NN and as Pikachu with 10-NN.")

# SHOWING THE SCATTER PLOT
plt.legend()
plt.show()
