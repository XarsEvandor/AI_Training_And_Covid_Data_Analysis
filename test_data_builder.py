import utils
import random


def input_file_creator():
    file_name = "test_data.txt"
    usr_choice = True

    while True:
        print("\n\nPlease enter the data file name.")
        file_name = input("Input: ")

        if file_name == '':
            print("No file name entered.\nWas that intentional?")
            usr_choice = utils.trap(2)

            if not usr_choice:
                print("Please try again.")
                continue
            else:
                print("Default file name (test_data.txt) will be used.")
                file_name = "test_data.txt"

        elif file_name[-4] != ".":
            print(
                "You seem to have forgotten the file extension (i.e .txt).\nWas that intentional?")
            usr_choice = utils.trap(2)

            if not usr_choice:
                print("Please try again.")
                continue

        try:
            out_file = open(file_name, "x")
        except FileExistsError:
            print("File already exists. Want to overwrite it?")
            usr_choice = utils.trap(2)

            if not usr_choice:
                print("Please enter a different file name.")
                while True:
                    print("\nPlease enter the data file name.")
                    file_name = input("Input: ")

                    if file_name == '':
                        print("No file name entered.\nWas that intentional?")
                        usr_choice = utils.trap(2)

                        if not usr_choice:
                            print("Please try again.")
                            continue
                        else:
                            print(
                                "Default file name (test_data.txt) will be used.")
                            file_name = "test_data.txt"

                    elif file_name[-4] != ".":
                        print(
                            "You seem to have forgotten the file extension (i.e .txt).\nWas that intentional?")
                        usr_choice = utils.trap(2)

                        if not usr_choice:
                            print("Please try again.")
                            continue

                    try:
                        out_file = open(file_name, "x")
                        break
                    except FileExistsError:
                        print("File already exists. Want to overwrite it?")
                        usr_choice = utils.trap(2)

                    if not usr_choice:
                        print("Please try again.")
                    else:
                        break

        break

    out_file = open(file_name, "w")

    return(out_file, file_name)


def data_maker(epochs):

    # We have spaces between the one hot digits so we can split them into different columns later.
    hot_encoding_dict = {0: '0 0 0 0 0 0 0 1',
                         1: '0 0 0 0 0 0 1 0', 2: '0 0 0 0 0 1 0 0', 3: '0 0 0 0 1 0 0 0',
                         4: '0 0 0 1 0 0 0 0', 5: '0 0 1 0 0 0 0 0', 6: '0 1 0 0 0 0 0 0',
                         7: '1 0 0 0 0 0 0 0'}

    out_file, file_name = input_file_creator()

    for i in range(0, epochs):
        out_file.write(hot_encoding_dict.get(random.randint(0, 7)) + "\n")

    out_file.close()

    return(file_name)
