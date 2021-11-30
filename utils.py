# PROJECT TITLE: User Error Trapping
# AUTHOR NAME: GIORGOS-PANAGIOTIS KATSONIS
# PURPOSE OF PROJECT: A versitile function for user input error trapping
# VERSION or DATE: 5/12/2021
# AUTHORS: giorgos_katsonis@hotmail.com
# COPYRIGHT INFORMATION:  Content is copyright © Open Source Guides authors, released under CC-BY-4.0.

def trap(type=1, accept_empty=0, upper_lim=None, lower_lim=None):

    user_in = 0

    if type == 1:
        # Error Trapping begins
        trap_bool = False
        # Accept_empty dictates if the user will be allowed to enter an empty string. Usefull for having enter be a sentinel value.
        if accept_empty:
            while not trap_bool:
                user_in = input("Input: ")

                if user_in == "":
                    return(user_in)
                else:
                    try:
                        user_in = int(user_in)
                        trap_bool = True
                    except ValueError:
                        print("Integer required. Please try again.")
        else:
            while not trap_bool:
                try:
                    user_in = int(input("Input: "))
                    trap_bool = True
                except ValueError:
                    print("Integer required. Please try again.")

        # Checking for margins
        if not (upper_lim is None and lower_lim is None):
            while True:
                if upper_lim is not None and user_in < lower_lim:
                    print("Violated lower margin. Please try again")

                    # Added failsafe for especially dense users
                    trap_bool = False
                    if accept_empty:
                        while not trap_bool:
                            user_in = input("Input: ")

                            if user_in == "":
                                return(user_in)
                            else:
                                try:
                                    user_in = int(user_in)
                                    trap_bool = True
                                except ValueError:
                                    print("Integer required. Please try again.")
                    else:
                        while not trap_bool:
                            try:
                                user_in = int(input("Input: "))
                                trap_bool = True
                            except ValueError:
                                print("Integer required. Please try again.")

                elif lower_lim is not None and user_in > upper_lim:
                    print("Violated upper margin. Please try again")

                    # Added failsafe for especially dense users
                    trap_bool = False
                    if accept_empty:
                        while not trap_bool:
                            user_in = input("Input: ")

                            if user_in == "":
                                return(user_in)
                            else:
                                try:
                                    user_in = int(user_in)
                                    trap_bool = True
                                except ValueError:
                                    print("Integer required. Please try again.")
                    else:
                        while not trap_bool:
                            try:
                                user_in = int(input("Input: "))
                                trap_bool = True
                            except ValueError:
                                print("Integer required. Please try again.")
                else:
                    break
    elif type == 0:
        # Error Trapping begins
        trap_bool = False
        if accept_empty:
            while not trap_bool:
                user_in = input("Input: ")

                if user_in == "":
                    return(user_in)
                else:
                    try:
                        user_in = float(user_in)
                        trap_bool = True
                    except ValueError:
                        print("Float required. Please try again.")
        else:
            while not trap_bool:
                try:
                    user_in = float(input("Input: "))
                    trap_bool = True
                except ValueError:
                    print("Float required. Please try again.")

        # Checking for margins
        if not (upper_lim is None and lower_lim is None):
            while True:
                if upper_lim is not None and user_in < lower_lim:
                    print("Violated lower margin. Please try again")

                    # Added failsafe for especially dense users
                    trap_bool = False
                    if accept_empty:
                        while not trap_bool:
                            user_in = input("Input: ")

                            if user_in == "":
                                return(user_in)
                            else:
                                try:
                                    user_in = int(user_in)
                                    trap_bool = True
                                except ValueError:
                                    print("Integer required. Please try again.")
                    else:
                        while not trap_bool:
                            try:
                                user_in = int(input("Input: "))
                                trap_bool = True
                            except ValueError:
                                print("Integer required. Please try again.")
                elif lower_lim is not None and user_in > upper_lim:
                    print("Violated upper margin. Please try again")

                    # Added failsafe for especially dense users
                    trap_bool = False
                    if accept_empty:
                        while not trap_bool:
                            user_in = input("Input: ")

                            if user_in == "":
                                return(user_in)
                            else:
                                try:
                                    user_in = int(user_in)
                                    trap_bool = True
                                except ValueError:
                                    print("Integer required. Please try again.")
                    else:
                        while not trap_bool:
                            try:
                                user_in = int(input("Input: "))
                                trap_bool = True
                            except ValueError:
                                print("Integer required. Please try again.")
                else:
                    break
    else:
        # Error Trapping begins
        while True:
            user_in = input("Input (Y/N): ")

            if len(user_in) > 1:
                print(
                    "Please enter Y for yes or N for no. You entered too many characters.")

            elif user_in.find("Y") == 0 or user_in.find("y") == 0:
                return True

            elif user_in.find("N") == 0 or user_in.find("n") == 0:
                return False

            else:
                print(
                    "Please enter Y for yes or N for no. Ensure your keyboard is in English.")

    return user_in
