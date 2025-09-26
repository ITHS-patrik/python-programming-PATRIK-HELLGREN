import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

current_dir = Path(__file__).parent
training_path = current_dir/"datapoints.txt"
test_path = current_dir/"testpoints.txt"

def read_and_clean_data(data_path):
    """ Reading data from the source files and finally converting it to NumPy arrays. """
    cleaned_data = []
    
    try:
        with open(data_path, "r") as txt:
            next(txt)

            for data in txt:
                if data_path == training_path:
                    split_data = data.split(",")
                    data_points = [float(d.strip().replace("(", "").replace(")", "")) for d in split_data]
                else:
                    data = data.split(" ", 1)[1]
                    data_points = [float(d.strip().replace("(", "").replace(")", "")) for d in data.split(",")]
                cleaned_data.append(data_points)
    except OSError as e:
        print(f'An error occurred while reading the training data file: {e}.\n'f"Please check your path/file name.\nClosing the application.")
        sys.exit(1)

    if data_path == training_path:
        train_x, train_y, train_z = zip(*cleaned_data)
        train_x, train_y, train_z = np.array(train_x), np.array(train_y), np.array(train_z)

        return cleaned_data, train_x, train_y, train_z
    else:
        test_x, test_y = zip(*cleaned_data)
        test_x, test_y = np.array(test_x), np.array(test_y)

        return cleaned_data, test_x, test_y

def assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y, input_x=None, input_y=None, print_sample_classification=False):
    """ Assigning nearest neighbour(s) for input data and both sets of test data, then printing the specific sample classification. """

    training_points = np.column_stack((train_x, train_y))
    training_labels = np.array(train_z)
    test_points = np.column_stack((test_x, test_y))
    predictions = []

    predictions = training_labels[np.argmin(np.linalg.norm(training_points[:, None, :] - test_points[None, :, :], axis=2), axis=0)] # Calculating the distance between each test and training point and making a prediction.

    if print_sample_classification: # only printing this for testpoints.txt and not for the 50 test points in datapoints.txt to not crowd the terminal.
        for i in range(len(test_x)):
            print(f"Sample with (width, height): ({test_x[i]}, {test_y[i]}) classified as {"Pichu" if predictions[i] == 0 else "Pikachu"}")

    pokemon_1nn = pokemon_10nn = nearest_ids = None

    if input_x is not None and input_y is not None: # making predictions based on user input (both 1nn and 10nn).
        distances = np.linalg.norm(training_points - np.array([input_x, input_y]), axis=1)
        
        prediction_label_1nn = training_labels[np.argmin(distances)] # 1-NN
        pokemon_1nn = "Pichu" if prediction_label_1nn == 0 else "Pikachu"
        print(f"Your input ({input_x}, {input_y}) classifies as {pokemon_1nn} with 1-NN.")

        nearest_ids = np.argsort(distances)[:10] # 10-NN
        label_count = np.bincount(training_labels[nearest_ids].astype(int)).argmax()
        pokemon_10nn = "Pichu" if label_count == 0 else "Pikachu"
        print(f"Your input ({input_x}, {input_y}) classifies as {pokemon_10nn} with 10-NN.")

    return predictions, training_points, training_labels, pokemon_1nn, pokemon_10nn, nearest_ids

def plot_data(train_z, test_x, test_y, predictions, pokemon_1nn, pokemon_10nn, input_x, input_y, nearest_ids, training_points, collected_accuracy):
    """ Plotting all the training, test and input data and also the accuracy of the model (in a separate plot). """
    fig, (f1, f2) = plt.subplots(1, 2, figsize=(13,6))
    fig.suptitle("Lab 2 - Pichu or Pikachu?", fontsize=16, fontweight="bold")
    fig.set_facecolor("lightgrey")
    
    # Training data
    f1.scatter(training_points[train_z == 0, 0], training_points[train_z == 0, 1], color="green", alpha=0.6, edgecolors="black", label="Pichu", marker="o")
    f1.scatter(training_points[train_z == 1, 0], training_points[train_z == 1, 1], color="yellow", alpha=0.6, edgecolors="black", label="Pikachu", marker="o")
    
    # Test data
    f1.scatter(test_x[predictions == 0], test_y[predictions == 0], color="green", edgecolors="black", label="Pichu (test)", marker="^")
    f1.scatter(test_x[predictions == 1], test_y[predictions == 1], color="yellow", edgecolors="black", label="Pikachu (test)", marker="^")

    # Input data
    f1.scatter(input_x, input_y, color="red", edgecolors="black", label="Pichu (user input 1-NN)" if pokemon_1nn == "Pichu" else "Pikachu (user input 1-NN)", marker="*", s=125)
    f1.scatter(input_x, input_y, color="red", edgecolors="black", label="Pichu (user input 10-NN)" if pokemon_10nn == "Pichu" else "Pikachu (user input 10-NN)", marker="*", s=125)
    for i in nearest_ids: # change the edgecolor for 10NN data points.
       f1.scatter(training_points[i, 0], training_points[i, 1], color="None", label="10 nearest neighbours" if i == nearest_ids[-1] else "", edgecolors="purple", linewidths=1.5, marker="o")

    f1.set_xlabel("Width (cm)")
    f1.set_ylabel("Height (cm)")
    f1.set_title("Data points and nearest neighbour(s)", fontsize=12)
    f1.grid(True)
    f1.legend()

    f2.plot(range(1, len(collected_accuracy)+1), np.array(collected_accuracy)*100, marker="^", label="Accuracy per loop", color="purple")
    f2.axhline(y = np.mean(collected_accuracy)*100, linestyle="--", label=f"Mean accuracy: {np.mean(collected_accuracy):.1%}", color="green")
    f2.set_xlabel("Loop (no.)")
    f2.set_ylabel("Accuracy (%)")
    f2.set_title("The accuracy of the model", fontsize=12)
    f2.set_ylim(83, 100)
    f2.grid(True)
    f2.legend()

    plt.tight_layout()
    plt.show()

def user_input(training_points, training_labels):
    """ Checking the user input data for errors and if none sending it forward to assign nearest neighbours. """

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
                print(f'Your input "({input_x}, {input_y})" contains invalid characters! Only use integers or floats > 0.\nTry again.')

        else:
            print(f"Thank you!")
            predictions, training_points, training_labels, pokemon_1nn, pokemon_10nn, nearest_ids = assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y, input_x, input_y)
            break

    return input_x, input_y, predictions, training_points, training_labels, pokemon_1nn, pokemon_10nn, nearest_ids

def split_data_points(total_x, total_y, total_z):
    """ Splitting the original datapoints into both training (100) and test (50) points equally divided amongst Pichu and Pikachu. Also shuffling all data for use in calculate_accuracy. """
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
    """ Looping through the 50 test data points 10 times, calculating the accuracy of every loop and then printing the mean accuracy of the model. """
    collected_accuracy = []
    
    for i in range(10):
        train_x, train_y, train_z, test_x, test_y, test_z = split_data_points(total_x, total_y, total_z)
        predictions, *_ = assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y)
        
        correct_predictions = np.sum(predictions == test_z) # True positives & True negatives
        accuracy = np.mean(predictions == test_z)
        collected_accuracy.append(accuracy)
        total_predictions = len(predictions) # True positives, True negatives, False positives & False negatives (total)
        print(f"The accuracy of this loop ({i+1}) is: {accuracy:.2%}. The model got it right {correct_predictions} out of {total_predictions} times.")

    mean_accuracy = np.mean(collected_accuracy)
    print(f"The mean accuracy of these ten loops were: {mean_accuracy:.1%}.")

    return collected_accuracy

_, train_x, train_y, train_z = read_and_clean_data(training_path)
_, test_x, test_y = read_and_clean_data(test_path)
predictions, training_points, training_labels, *_ = assign_nearest_neighbour(train_x, train_y, train_z, test_x, test_y, print_sample_classification=True)
input_x, input_y, _, _, _, pokemon_1nn, pokemon_10nn, nearest_ids = user_input(training_points, training_labels)
_, total_x, total_y, total_z = read_and_clean_data(training_path)
collected_accuracy = calculate_accuracy(total_x, total_y, total_z)
plot_data(train_z, test_x, test_y, predictions, pokemon_1nn, pokemon_10nn, input_x, input_y, nearest_ids, training_points, collected_accuracy)
