import sys
import subprocess


# use pip to install numpy:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'numpy'])

# use pip to install matplotlib:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'matplotlib'])

# use pip to install pandas:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'pandas'])

# use pip to install sklearn:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'sklearn'])




from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import training_data_builder
import test_data_builder
import matplotlib.pyplot as plt
import utils
import numpy as np
import warnings

warnings.filterwarnings("ignore")

epochs = 500
mid_layer_size = 15


def option1():
    print("Enter the size of the hidden layer. Please limit your choice to integers between 2 and 99.")
    print("Pressing enter without entering a value will select the default value of 15.")
    global mid_layer_size
    mid_layer_size = utils.trap(1, 1, 99, 2)

    if mid_layer_size == '':
        mid_layer_size = 15


def option2():
    global epochs
    print("Enter the number of epochs. Please limit your choice to integers between 50 and 10.000.")
    print("Pressing enter without entering a value will select the default value of 500.")
    epochs = utils.trap(1, 1, 10000, 50)

    if epochs == '':
        epochs = 500

    file_name = training_data_builder.data_maker(epochs)

    print("\n\nPlease select the training step funtion you wish to use. Pressing enter without selecting will use the default function.\n")
    print("1. Identity Function: f(x) = x ")
    print("2. Logistic sigmoid function: f(x) = 1 / (1 + exp(-x))")
    print("3. Hyperbolic tan function: f(x) = tanh(x)")
    print("Default. Rectified linear unit function: f(x) = max(0, x)")

    function_choice = utils.trap(1, 1, 3, 1)

    if function_choice == '':
        step_function = 'relu'
    else:
        if function_choice == 1:
            step_function = 'identity'
        elif function_choice == 2:
            step_function = 'logistic'
        elif function_choice == 3:
            step_function = 'tanh'

    print("\n\nPlease select the solver for weight optimization you wish to use. Pressing enter without selecting will use the default function.\n")
    print("1. lbfgs: an optimizer in the family of quasi-Newton methods")
    print("2. sgd: stochastic gradient descent")
    print("Default. adam: a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba")
    print("""\
        \n\nNote: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) 
        in terms of both training time and validation score. 
        For small datasets, however, ‘lbfgs’ can converge faster and perform better.
        If sgd is chosen, adaptive learning will be applied.
        """)

    solver_choice = utils.trap(1, 1, 2, 1)

    if solver_choice == '':
        solver = 'adam'
    else:
        if solver_choice == 1:
            solver = 'lbfgs'
        elif solver_choice == 2:
            solver = 'sgd'

    # Assign column names to the dataset
    names = ['one_hot', 'is_odd', 'larger_than_3']

    # Read dataset to pandas dataframe. We set the one-hot column as a string so that we dont lose the leading zeros.
    in_data = pd.read_csv(file_name, names=names, dtype={'one-hot': 'string'})
    in_data[['dig0', 'dig1', 'dig2', 'dig3', 'dig4', 'dig5', 'dig6', 'dig7']
            ] = in_data.one_hot.str.split(" ", expand=True)

    # Let us peek at the top of the data set, as imported in the in_data dataframe
    print("\nPrinting the first few lines of the data set:\n", in_data.head())

    # Assign data to X variable
    X = in_data.iloc[:, 3:11]

    # Assign data to y variable
    y = in_data.iloc[:, 1:3]

    # Processing data
    le = preprocessing.LabelEncoder()

    y = y.apply(le.fit_transform)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Training. Adaptive learning reduces the learning rate if the error change remains below the defult tolerance twice in a row.
    # We use n_iter_no_change to increase the number of epochs that can pass with the error not changing above the tolerance without terminating the training early.
    # For this problem, around 500 epochs are the limit of training for a tolerance of 1e-4.
    mlp = MLPClassifier(hidden_layer_sizes=(mid_layer_size),
                        max_iter=epochs, activation=step_function, learning_rate='adaptive', n_iter_no_change=10000, solver=solver)
    mlp.fit(X_train, y_train)

    # Testing
    predictions = mlp.predict(X_test)

    # Displaying test results
    training_progress_file = open("training_progress.txt", "w")

    training_progress_file.write("{:<5} {:>10}".format(
        "Epochs", "Error\n"))
    training_progress_file.write(
        "-" * len("{:<5} {:>10}".format("Epochs", "Error")))
    training_progress_file.write("\n")

    for i in range(0, epochs-1):
        training_progress_file.write(
            "{:<5} {:>10} {:}".format(i, mlp.loss_curve_[i], "\n"))

    print('Now printing the confusion matrix...\n')
    print(confusion_matrix(y_test.values.argmax(
        axis=1), predictions.argmax(axis=1)))
    print('\nNow printing the classification report...\n')
    print(classification_report(y_test.values.argmax(
        axis=1), predictions.argmax(axis=1)))

    print("\nTraining progress was saved in training_progress.txt")

    training_progress_file.close()
    return(mlp)


def option3(mlp):
    file_name = test_data_builder.data_maker(epochs)

    training_out_file = open("training_output.txt", "w")

    # Assign column names to the dataset
    names = ['one_hot']

    # Read dataset to pandas dataframe. We set the one-hot column as a string so that we dont lose the leading zeros.
    in_data = pd.read_csv(file_name, names=names, dtype={'one-hot': 'string'})
    in_data[['dig0', 'dig1', 'dig2', 'dig3', 'dig4', 'dig5', 'dig6', 'dig7']
            ] = in_data.one_hot.str.split(" ", expand=True)

    # Assign data from first four columns to X variable
    X = in_data.iloc[:, 1:9]

    print(X)

    predictions = mlp.predict(X)

    print("\n\n")

    # Turn the dataframe into a 2d Numpy array.
    x_matrix = X.to_numpy()

    # Extract data, format it and then write on file.
    for i in range(0, len(predictions) - 1):
        y_one_hot = ''

        y_is_odd, y_is_over_3 = str(predictions[i]).replace(
            "[", "").replace("]", "").split(" ")

        for k in x_matrix[i]:
            y_one_hot += str(k)

        print(y_one_hot + "," + y_is_odd + "," + y_is_over_3)
        training_out_file.write(
            (y_one_hot + "," + y_is_odd + "," + y_is_over_3))
        training_out_file.write("\n")

    print("\n\nData has been saved on training_output.txt")
    training_out_file.close()


def option4(mlp):
    fig = plt.figure(figsize=(10, 6))

    # Set up the chart.
    error_figure = fig.add_subplot()
    error_figure.set(Title="Error/Epochs",
                     xlabel="Epochs", ylabel="Error")

    error_figure.plot(range(epochs), mlp.loss_curve_)

    plt.tight_layout()
    plt.show()


def menu():
    print("\n\nWelcome to the AVERAGE MLP PROGRAM!")

    while True:

        print(
            "\n\n==========================================================================")
        print("Option 1: Enter size of the middle layer.")
        print("Option 2: Initiate training pass.")
        print("Option 3: LOCKED")
        print("Option 4: LOCKED")
        print("Option 5: Exit.")
        print(
            "==========================================================================\n\n")

        choice = utils.trap(1, 0, 5, 1)

        if choice == 1:
            option1()
        elif choice == 2:
            mlp = option2()
            break
        elif choice == 3:
            print("REQUIRES OPTION 2")
        elif choice == 4:
            print("REQUIRES OPTION 2")
        else:
            print("Are you sure you want to exit?")
            bool_choice = utils.trap(2)

            if bool_choice:
                print("Thank you for using the AVERAGE MLP PROGRAM!")

                print("{:>80}".format("...have a nice day!"))
                exit()

    while True:

        print(
            "\n\n==========================================================================")
        print("Option 1: Enter size of the middle layer.")
        print("Option 2: Initiate training pass.")
        print("Option 3: Classify test data.")
        print("Option 4: Display training result graphics.")
        print("Option 5: Exit.")
        print(
            "==========================================================================\n\n")

        choice = utils.trap(1, 0, 5, 1)

        if choice == 1:
            option1()
        elif choice == 2:
            mlp = option2()
        elif choice == 3:
            option3(mlp)
        elif choice == 4:
            option4(mlp)
        else:
            print("Are you sure you want to exit?")
            bool_choice = utils.trap(2)

            if bool_choice:
                print("Thank you for using the AVERAGE MLP PROGRAM!")

                print("{:>80}".format("...have a nice day!"))
                break


menu()
