import matplotlib.pyplot as plt
import numpy as np
import sys

data_path_training = "Labs/02_Programmering_inom_Python/Laboration_2/datapoints.txt"
data_path_test = "Labs/02_Programmering_inom_Python/Laboration_2/testpoints.txt"
cleaned_data = []

# FUNCTION FOR READING & CLEANING THE DATA
def read_and_clean_data(data_path):
    
    cleaned_data.clear()
    with open(data_path, "r") as txt:
        next(txt)

        for data in txt:
            try:
                if data_path == data_path_training:
                    split_data = data.split(",")
                    data_points = [float(d.strip().replace("(", "").replace(")", "")) for d in split_data]
                else:
                    data = data.split(" ", 1)[1]
                    data_points = [float(d.strip().replace("(", "").replace(")", "")) for d in data.split(",")]
                cleaned_data.append(data_points)

            except OSError as e:
                print(f"An error occurred ({e}) while reading the training data file.\n"
                f"Please check your path/file name ({data_path}), network and permissions."
                ""
                "Closing the application.")
                sys.exit(1)

    return cleaned_data

# FUNCTION FOR SEPARATING X, Y AND Z VALUES & CONVERTING THEM TO NUMPY ARRAYS
def split_and_convert_xyz(data_path):

    if data_path == data_path_training:
        train_x, train_y, train_z = zip(*read_and_clean_data(data_path))
        train_x, train_y, train_z = np.array(train_x), np.array(train_y), np.array(train_z)

        return train_x, train_y, train_z
    else:
        test_x, test_y = zip(*read_and_clean_data(data_path))
        test_x, test_y = np.array(test_x), np.array(test_y)

        return test_x, test_y

# FUNCTION FOR ASSIGNING NEAREST NEIGHBOUR
def assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y):

    training_points = np.column_stack((train_x, train_y))
    training_labels = train_z
    test_points = np.column_stack((test_x, test_y))

    predictions = []

    for te_point in test_points:
        diff_xy = training_points[:, np.newaxis, :] - te_point
        distances = np.sqrt(np.sum(diff_xy ** 2, axis=2))
        nearest_id = np.argmin(distances, axis=0)
        prediction_label = training_labels[nearest_id]
        predictions.append(prediction_label)

    predictions = np.array(predictions).ravel()

    return predictions, training_points, training_labels

# FUNCTION FOR TEST DATA CLASSIFICATION (1-NN)
def pokemon_classification_1NN(test_data_predictions, test_x, test_y):
    
    for i in range(len(test_x)):
        pokemon_1NN = "Pichu" if test_data_predictions[i] == 0 else "Pikachu"
        print(f"Sample with (width, height): ({test_x[i]}, {test_y[i]}) classified as {pokemon_1NN}")

# FUNCTION FOR PLOTTING ALL DATA
def plot_data(**kwargs):
    
    # Training data
    if all(k in kwargs for k in ("train_x", "train_y", "train_z")):
        train_x = kwargs["train_x"]
        train_y = kwargs["train_y"]
        train_z = kwargs["train_z"]
        plt.scatter(train_x[train_z == 0], train_y[train_z == 0], color="green", alpha=0.6, edgecolors="black", label="Pichu", marker="o")
        plt.scatter(train_x[train_z == 1], train_y[train_z == 1], color="yellow", alpha=0.6, edgecolors="black", label="Pikachu", marker="o")

    # Test data
    if all(k in kwargs for k in ("test_x", "test_y", "predictions")):
        test_x = kwargs["test_x"]
        test_y = kwargs["test_y"]
        predictions = kwargs["predictions"]
        plt.scatter(test_x[predictions == 0], test_y[predictions == 0], color="green", edgecolors="black", label="Pichu (test)", marker="^")
        plt.scatter(test_x[predictions == 1], test_y[predictions == 1], color="yellow", edgecolors="black", label="Pikachu (test)", marker="^")

    # Input data 1-NN
    if all(k in kwargs for k in ("pokemon_1NN", "refined_input_x", "refined_input_y")):
        pokemon_1NN = kwargs["pokemon_1NN"]
        refined_input_x = kwargs["refined_input_x"]
        refined_input_y = kwargs["refined_input_y"]
        if pokemon_1NN == "Pichu":
            plt.scatter(refined_input_x, refined_input_y, color="red", edgecolors="black", label="Pichu (user input 1-NN)", marker="*", s=175)
        else:
            plt.scatter(refined_input_x, refined_input_y, color="red", edgecolors="black", label="Pikachu (user input 1-NN)", marker="*", s=175)

    # Input data 10-NN
    if all(k in kwargs for k in ("pokemon_10NN", "refined_input_x_10NN", "refined_input_y_10NN")):
        pokemon_10NN = kwargs["pokemon_10NN"]
        refined_input_x_10NN = kwargs["refined_input_x_10NN"]
        refined_input_y_10NN = kwargs["refined_input_y_10NN"]
        if pokemon_10NN == "Pichu":
            plt.scatter(refined_input_x_10NN, refined_input_y_10NN, color="red", edgecolors="black", label="Pichu (user input 10-NN)", marker="*", s=175)
        else:
            plt.scatter(refined_input_x_10NN, refined_input_y_10NN, color="red", edgecolors="black", label="Pikachu (user input 10-NN)", marker="*", s=175)

    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Pichu eller Pikachu?")
    plt.grid(True)
    plt.legend()

# FUNCTION FOR USER INPUT (1-NN) & ERROR HANDLING
def user_input_1NN_Error_handling(training_points, training_labels):

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
            input_diff_xy = training_points - input_data_point
            distances = np.sqrt(np.sum(input_diff_xy ** 2, axis=1))
            nearest_id = np.argmin(distances)
            prediction_label = training_labels[nearest_id]
            pokemon_1NN = "Pichu" if prediction_label == 0 else "Pikachu"

            print(f"Thank you!")
            print(f"Your input ({refined_input_x}, {refined_input_y}) classifies as {pokemon_1NN} with 1-NN.")
            break

    return refined_input_x, refined_input_y, distances, pokemon_1NN

# FUNCTION FOR USER INPUT (10-NN)
def user_input_10NN(refined_input_x, refined_input_y, distances):

    refined_input_x_10NN = refined_input_x
    refined_input_y_10NN = refined_input_y
    #input_data_point_10NN = np.array([refined_input_x_10NN, refined_input_y_10NN]) <-- används ej i print längst ned. ska den användas?

    ten = 10
    nearest_ids = np.argsort(distances)[:ten]
    nearest_ids_labels = training_labels[nearest_ids]

    label_count = np.bincount(nearest_ids_labels.astype(int))
    prediction_label = np.argmax(label_count)

    pokemon_10NN = "Pichu" if prediction_label == 0 else "Pikachu"

    print(f"Your input ({refined_input_x_10NN}, {refined_input_y_10NN}) classifies as {pokemon_10NN} with 10-NN.")

    return pokemon_10NN, refined_input_x_10NN, refined_input_y_10NN

# FUNCTION FOR PRINTING A REFLECTION ABOUT 1-NN VS. 10-NN
def reflection():
    reflection_x = 21.8
    reflection_y = 33.5
    print(f"Reflection: when the user types in e.g. ({reflection_x}, {reflection_y}) the model classifies the pokemon as Pikachu with 1-NN and as Pichu with 10-NN.")

# PROCESS THE TRAINING DATA
read_and_clean_data(data_path_training)
train_x, train_y, train_z = split_and_convert_xyz(data_path_training)
plot_data(train_x=train_x, train_y=train_y, train_z=train_z)

# PROCESS THE TEST DATA
read_and_clean_data(data_path_test)
test_x, test_y = split_and_convert_xyz(data_path_test)
predictions, training_points, training_labels = assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y)
pokemon_classification_1NN(predictions, test_x, test_y)
plot_data(test_x=test_x, test_y=test_y, predictions=predictions)

# PROCESS THE USER INPUT DATA
refined_input_x, refined_input_y, distances, pokemon_1NN = user_input_1NN_Error_handling(training_points, training_labels)
pokemon_10NN, refined_input_x_10NN, refined_input_y_10NN = user_input_10NN(refined_input_x, refined_input_y, distances)
plot_data(pokemon_1NN=pokemon_1NN, refined_input_x=refined_input_x, refined_input_y=refined_input_y)
plot_data(pokemon_10NN=pokemon_10NN, refined_input_x_10NN=refined_input_x_10NN, refined_input_y_10NN=refined_input_y_10NN)
reflection()

# SHOWING THE SCATTER PLOT
plt.show()
