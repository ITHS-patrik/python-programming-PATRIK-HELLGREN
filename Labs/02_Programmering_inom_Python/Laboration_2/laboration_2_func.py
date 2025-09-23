import matplotlib.pyplot as plt
import numpy as np
import sys

data_path_training = "Labs/02_Programmering_inom_Python/Laboration_2/datapoints.txt"
data_path_test = "Labs/02_Programmering_inom_Python/Laboration_2/testpoints.txt"

# FUNCTION FOR READING & CLEANING THE DATA
def read_and_clean_data(data_path):
    cleaned_data = []
    
    try:
        with open(data_path, "r") as txt:
            next(txt)

            for data in txt:
                if data_path == data_path_training:
                    split_data = data.split(",")
                    data_points = [float(d.strip().replace("(", "").replace(")", "")) for d in split_data]
                else:
                    data = data.split(" ", 1)[1]
                    data_points = [float(d.strip().replace("(", "").replace(")", "")) for d in data.split(",")]
                cleaned_data.append(data_points)

    except OSError as e:
        print(f'An error occurred while reading the training data file: {e}.\n'f"Please check your path/file name, network and permissions.\nClosing the application.")
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

    for i in range(len(test_x)):
        pokemon_1NN = "Pichu" if predictions[i] == 0 else "Pikachu"
        print(f"Sample with (width, height): ({test_x[i]}, {test_y[i]}) classified as {pokemon_1NN}")

    return predictions, training_points, training_labels

# FUNCTION FOR PLOTTING ALL DATA
def plot_data(train_z, test_x, test_y, predictions, pokemon_1NN, pokemon_10NN, input_x, input_y, nearest_ids, training_points, collected_accuracy):
    
    # Training data
    plt.scatter(training_points[train_z == 0, 0], training_points[train_z == 0, 1], color="green", alpha=0.6, edgecolors="black", label="Pichu", marker="o")
    plt.scatter(training_points[train_z == 1, 0], training_points[train_z == 1, 1], color="yellow", alpha=0.6, edgecolors="black", label="Pikachu", marker="o")
    for i in nearest_ids:
        plt.scatter(training_points[i, 0], training_points[i, 1], color=f'{"green" if train_z[i] == 0 else "yellow"}', edgecolors="purple", linewidths=2, marker="o")

    # Test data
    plt.scatter(test_x[predictions == 0], test_y[predictions == 0], color="green", edgecolors="black", label="Pichu (test)", marker="^")
    plt.scatter(test_x[predictions == 1], test_y[predictions == 1], color="yellow", edgecolors="black", label="Pikachu (test)", marker="^")

    # Input data
    if pokemon_1NN == "Pichu":
        plt.scatter(input_x, input_y, color="red", edgecolors="black", label="Pichu (user input 1-NN)", marker="*", s=175)
    else:
        plt.scatter(input_x, input_y, color="red", edgecolors="black", label="Pikachu (user input 1-NN)", marker="*", s=175)

    if pokemon_10NN == "Pichu":
        plt.scatter(input_x, input_y, color="red", edgecolors="black", label="Pichu (user input 10-NN)", marker="*", s=175)
    else:
        plt.scatter(input_x, input_y, color="red", edgecolors="black", label="Pikachu (user input 10-NN)", marker="*", s=175)

    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Pichu eller Pikachu?")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(range(1, len(collected_accuracy)+1), np.array(collected_accuracy)*100, marker="^", label="Accuracy per loop", color="purple")
    plt.axhline(y = np.mean(collected_accuracy)*100, linestyle="--", label=f"Mean accuracy: {np.mean(collected_accuracy):.1%}", color="green")
    plt.xlabel("Loop (no.)")
    plt.ylabel("Accuracy (%)")
    plt.title("The accuracy of the Pichu/Pikachu model")
    plt.ylim(85, 100)
    plt.grid(True)
    plt.legend()
    plt.show()

# FUNCTION FOR USER INPUT
def user_input(training_points, training_labels):

    while True:
        try:
            input_x = float(input("Please input the x value (width) for your test data point: "))
            input_y = float(input("Please input the y value (height) for your test data point: "))

            if input_x <= 0 or input_y <= 0:
                raise ValueError("NegativeValue")
            
        except ValueError as e:
            if str(e) == "NegativeValue":
                print(f"Null or negative values are not accepted.\nYour input was: ({input_x}, {input_y}).\nTry again.")
            else:
                print(f'Your input "({input_x}, {input_y})" contains invalid characters! Only use floats > 0.\nTry again.')

        else:
            print(f"Thank you!")
            input_data_point = np.array([input_x, input_y])
            distances = np.sqrt(np.sum((training_points - input_data_point) ** 2, axis=1))
            
            nearest_id = np.argmin(distances) # 1-NN
            prediction_label_1NN = training_labels[nearest_id]
            pokemon_1NN = "Pichu" if prediction_label_1NN == 0 else "Pikachu"
            print(f"Your input ({input_x}, {input_y}) classifies as {pokemon_1NN} with 1-NN.")

            nearest_ids = np.argsort(distances)[:10] # 10-NN
            nearest_ids_labels = training_labels[nearest_ids]
            label_count = np.bincount(nearest_ids_labels.astype(int))
            prediction_label_10NN = np.argmax(label_count)
            pokemon_10NN = "Pichu" if prediction_label_10NN == 0 else "Pikachu"
            print(f"Your input ({input_x}, {input_y}) classifies as {pokemon_10NN} with 10-NN.")

            break

    return input_x, input_y, distances, pokemon_1NN, pokemon_10NN, nearest_ids

# FUNCTION FOR PRINTING A REFLECTION ABOUT 1-NN VS. 10-NN
def reflection():
    reflection_x = 21.8
    reflection_y = 33.5
    print(f"Reflection: when the user types in e.g. ({reflection_x}, {reflection_y}) the model classifies the pokemon as Pikachu with 1-NN and as Pichu with 10-NN.")

def split_data_points(total_x, total_y, total_z):
    training = 50
    test = 25

    pichu_data = np.where(total_z == 0)[0]
    np.random.shuffle(pichu_data)
    pichu_training_data = pichu_data[:training]
    pichu_test_data = pichu_data[training:training+test]

    pikachu_data = np.where(total_z == 1)[0]
    np.random.shuffle(pikachu_data)
    pikachu_training_data = pikachu_data[:training]
    pikachu_test_data = pikachu_data[training:training+test]

    new_training_data = np.concatenate([pichu_training_data, pikachu_training_data])
    new_test_data = np.concatenate([pichu_test_data, pikachu_test_data])

    train_x, train_y, train_z = total_x[new_training_data], total_y[new_training_data], total_z[new_training_data]
    test_x, test_y, test_z = total_x[new_test_data], total_y[new_test_data], total_z[new_test_data]

    return train_x, train_y, train_z, test_x, test_y, test_z

def calculate_accuracy(total_x, total_y, total_z):
    collected_accuracy = []

    for i in range(10):
        train_x, train_y, train_z, test_x, test_y, test_z = split_data_points(total_x, total_y, total_z)
        predictions, _, _ = assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y)
        
        correct_predictions = np.sum(predictions == test_z)
        accuracy = np.mean(predictions == test_z)
        collected_accuracy.append(accuracy)
        total_predictions = len(predictions)
        print(f"The accuracy of this loop ({i+1}) is: {accuracy:.2%}. The model got it right {correct_predictions} out of {total_predictions} times.")

    mean_accuracy = np.mean(collected_accuracy)
    print(f"The mean accuracy of these ten loops were: {mean_accuracy:.1%}.")

    return collected_accuracy

# PROCESS THE TRAINING DATA
read_and_clean_data(data_path_training)
train_x, train_y, train_z = split_and_convert_xyz(data_path_training)

# PROCESS THE TEST DATA
read_and_clean_data(data_path_test)
test_x, test_y = split_and_convert_xyz(data_path_test)
predictions, training_points, training_labels = assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y)

# PROCESS THE USER INPUT DATA
input_x, input_y, distances, pokemon_1NN, pokemon_10NN, nearest_ids = user_input(training_points, training_labels)
reflection()

# SPLIT ORIGINAL TRAINING DATA INTO NEW TRAINING AND TEST DATA
total_x, total_y, total_z = split_and_convert_xyz(data_path_training)
collected_accuracy = calculate_accuracy(total_x, total_y, total_z)

# PLOT DATA
plot_data(train_z, test_x, test_y, predictions, pokemon_1NN, pokemon_10NN, input_x, input_y, nearest_ids, training_points, collected_accuracy)
