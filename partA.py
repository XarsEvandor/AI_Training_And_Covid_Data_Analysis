import sys
import subprocess


# use pip to install numpy:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'numpy'])

# use pip to install matplotlib:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'matplotlib'])

import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Dictionary that translates the country codes to country names. Made by Konstantina Lambrou.
country_dict = {'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus', 'CZ': 'Czechia',
                'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain',
                'FI': 'Finland', 'FR': 'France', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
                'IS': 'Iceland', 'IT': 'Italy', 'LI': 'Liechtenstein', 'LT': 'Lithuania', 'LU': 'Luxembourg',
                'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland',
                'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia'}

# Dictionary that translates vaccine codes to vaccine names. Painstakingly made by me.
vaccine_dict = {'COM': 'Comirnaty - Pfizer/BioNTech', 'MOD': 'mRNA-1273 - Moderna',
                'BECNBG': 'BBIBV-CorV - CNBG', 'SIN': 'Coronavac - Sinovac', 'SPU': 'Sputnik V - GRI',
                'AZ': 'AZD1222 – AstraZeneca', 'JANSS': 'JNJ-78436735 - J&J/Janssen', 'UNK': 'UNKNOWN'}


def file_opener():
    # No matter the option, we need to ensure the input file exists
    # before continuing.
    while True:
        try:
            in_file = open("partA_input_data.txt")
            break
        except FileNotFoundError:
            print("partA_input_data.txt not found. Want to try again?")
            usr_choice = utils.trap(2)
            if not usr_choice:
                print("Have a good day! Goodbye.")
                print("Exiting..")
                exit()

    return in_file


def option1():
    in_file = file_opener()

    country_list = []
    date_list = []
    sorted_dates = []
    second_dose_list = []
    age_group_list = []

    # Using readlines we skip the header.
    # We only import the data we need from the file to avoid wasting resources.
    for line in in_file.readlines()[1:]:
        country_list.append(line.strip().split(",")[8])
        date_list.append(line.strip().split(",")[0])
        second_dose_list.append(line.strip().split(",")[3])
        age_group_list.append(line.strip().split(",")[9])

    # Import the country list into a set in order to have no repeating values.
    country_set = sorted(set(country_list))

    # Sort the date list in order to find the latest date no matter the version of the data.
    sorted_dates = sorted(date_list)
    latest_date = sorted_dates[-1]

    # Formatting used to create a tabular view of the data
    print("{:<10} {:<17} {:<10}".format("Country", "Vaccinated No.", "Week"))
    print("-----------------------------------\n")

    # For every distinct country..
    for country in country_set:
        sum = 0
        # For every entry in the data..
        for i in range(0, len(age_group_list) - 1):
            # If the age group is HWC and the date is equal to the latest date and the country is equal
            # to the distinct country of the current loop..
            if age_group_list[i].upper() == "HCW":
                if date_list[i] == latest_date:
                    if country_list[i] == country:
                        sum += int(second_dose_list[i])

        # print(country_dict.get(country) + ": " +
        #       str(sum) + "    " + latest_date)
        print("{:<15} {:<10} {:<10}".format(
            country_dict.get(country) + ": ", sum, latest_date[6:] + " " + latest_date[0:4]))

    in_file.close()


def option2():
    in_file = file_opener()

    country_list = []
    vaccine_list = []
    second_dose_list = []
    age_group_list = []

    # Using readlines we skip the header.
    # We only import the data we need from the file to avoid wasting resources.
    for line in in_file.readlines()[1:]:
        country_list.append(line.strip().split(",")[8])
        vaccine_list.append(line.strip().split(",")[10])
        second_dose_list.append(line.strip().split(",")[3])
        age_group_list.append(line.strip().split(",")[9])

    # Import the lists into a set in order to have no repeating values.
    country_set = sorted(set(country_list))
    vaccine_set = sorted(set(vaccine_list))

    print("\n\nPlease choose a country.")
    counter = 1

    # Display all the countries in tabular format
    for country in country_set:
        print("{:<4} {:}".format(
            str(counter) + ": ", country_dict.get(country)))
        counter += 1
    choice = utils.trap(1, 0, 29, 1)

    abreviated_country_list = list(country_set)

    # Display header
    print("\n\n")
    print(country_dict.get(abreviated_country_list[choice-1]))
    print("*" * len(country_dict.get(abreviated_country_list[choice-1])))

    print("{:<30} {:}".format("Vaccine", "Vaccination No."))
    print("-" * len("{:<30} {:}".format("Vaccine", "Vaccination No.")))

    # Display results
    for vaccine in vaccine_set:
        sum = 0
        for i in range(0, len(age_group_list) - 1):
            if age_group_list[i].upper() == "ALL":
                if country_list[i] == abreviated_country_list[choice - 1]:
                    if vaccine_list[i] == vaccine:
                        sum += int(second_dose_list[i])

        print("{:<30} {:}".format(vaccine_dict.get(
            vaccine, "UNLISTED") + ":", sum))

    in_file.close()


def option3():
    in_file = file_opener()

    country_list = []
    second_dose_list = []
    age_group_list = []
    date_list = []

    # Using readlines we skip the header.
    # We only import the data we need from the file to avoid wasting resources.
    for line in in_file.readlines()[1:]:
        country_list.append(line.strip().split(",")[8])
        second_dose_list.append(line.strip().split(",")[3])
        age_group_list.append(line.strip().split(",")[9])
        date_list.append(line.strip().split(",")[0])

    # Import the lists into a set in order to have no repeating values.
    country_set = sorted(set(country_list))
    date_set = sorted(set(date_list))

    print("\n\nPlease choose a country.")
    counter = 1

    # Display all the countries in tabular format
    for country in country_set:
        print("{:<4} {:}".format(
            str(counter) + ": ", country_dict.get(country)))
        counter += 1
    choice = utils.trap(1, 0, 29, 1)

    abreviated_country_list = list(country_set)

    # We want the data to be sorted on the dates. To do this we zip, sort and unzip.
    zipped_arrays = zip(date_list, age_group_list,
                        country_list, second_dose_list)
    sorted_date_list, sorted_age_group_list, sorted_country_list, sorted_second_dose_list = zip(
        *sorted(zipped_arrays))

    graph_dates = []
    graph_vaccinations = []
    fig = plt.figure(figsize=(10, 6))

    # Set up the chart.
    vaccination_figure = fig.add_subplot()
    vaccination_figure.set(Title="Health Care Worker vaccinations over time.\n" + country_dict.get(abreviated_country_list[choice - 1]),
                           xlabel="Weeks", ylabel="Vaccinations")

    # Display header
    print("\n\n")
    print(country_dict.get(abreviated_country_list[choice-1]))
    print("*" * len(country_dict.get(abreviated_country_list[choice-1])))

    print("{:<15} {:}".format("Date(YYYY/WW)", "Vaccinations"))
    print(
        "-" * len("{:<15} {:}".format("Date (YYYY/WW)" + ":", "Vaccinations")))

    # Plot and print relevant data.
    for date in date_set:
        sum = 0
        for i in range(0, len(sorted_age_group_list) - 1):
            if sorted_age_group_list[i].upper() == "HCW":
                if sorted_country_list[i] == abreviated_country_list[choice - 1]:
                    if sorted_date_list[i] == date:
                        sum += int(sorted_second_dose_list[i])

        print("{:<15} {:}".format(date[0:4]+" "+date[6:]+":", sum))
        graph_dates.append(date[6:])
        graph_vaccinations.append(sum)
        vaccination_figure.plot(graph_dates, graph_vaccinations)
        plt.pause(0.25)

    plt.tight_layout()
    plt.show()

    in_file.close()


def option4():
    in_file = file_opener()

    country_list = []
    second_dose_list = []
    age_group_list = []
    date_list = []
    population_list = []

    # Using readlines we skip the header.
    # We only import the data we need from the file to avoid wasting resources.
    for line in in_file.readlines()[1:]:
        country_list.append(line.strip().split(",")[8])
        second_dose_list.append(line.strip().split(",")[3])
        age_group_list.append(line.strip().split(",")[9])
        date_list.append(line.strip().split(",")[0])
        population_list.append(line.strip().split(",")[7])

    zipped_arrays = zip(population_list, country_list)
    sorted_population_list, sorted_country_list = list(
        zip(*sorted(zipped_arrays)))

    # Import the lists into a set in order to have no repeating values.
    country_set = set(sorted_country_list)

    population_dict = {}
    vaccinations_percentage_dict = {}

    # Avoiding going over the large lists over and over again by making a population dictionary.
    for country in country_set:
        population_dict[country] = [
            population_list[country_list.index(country)]]

    # calculate the percentages and create a vaccination percentage dictionary.
    for country in country_set:
        sum = 0
        for i in range(0, len(age_group_list) - 1):
            if age_group_list[i].upper() == "ALL":
                if country == country_list[i]:
                    sum += float(second_dose_list[i])

        vaccinations_percentage_dict[country] = [
            round(sum / float(population_dict.get(country)[0]), 3)]

    # Here we sort the dictionary. We use the sorted function, passing the values of the items in the dictionary as a list for the first arguement
    # and using a lambda function to specify sorting in respect of the values and not the keys.
    sorted_vaccinations_percentage_dict = dict(sorted(
        vaccinations_percentage_dict.items(), key=lambda x: x[1], reverse=True))

    file_name = "partA_output_data.txt"

    while True:
        try:
            out_file = open(file_name, "x")
        except FileExistsError:
            print("File already exists. Want to overwrite it?")
            usr_choice = utils.trap(2)

            if not usr_choice:
                print("Please enter a different file name.")
                while True:
                    print("\nPlease enter the output file name.")
                    file_name = input("Input: ")

                    if file_name[-4] != ".":
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

    out_file.write("{:<12} {:>10}".format(
        "Country", "Percentage of Vaccinations\n"))
    out_file.write(
        "-" * len("{:<12} {:>10}".format("Country", "Percentage of Vaccinations")))
    out_file.write("\n")

    for country in sorted_vaccinations_percentage_dict:
        out_file.write("{:<15} {:>10.2%} {:}".format(country_dict.get(
            country), sorted_vaccinations_percentage_dict.get(country)[0], "\n"))

    fig = plt.figure(figsize=(10, 6))

    countries = list(sorted_vaccinations_percentage_dict.keys())

    percentages = list(sorted_vaccinations_percentage_dict.values())

    for i in range(0, len(percentages)):
        percentages[i] = percentages[i][0]
        countries[i] = country_dict.get(countries[i])

    # Set up the chart.

    country_num = range(10)

    ax = plt.subplot(2, 1, 1)
    plt.ylim(0, 1)
    top_countries_figure = plt.bar(
        countries[:10], percentages[:10])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title("Top 10 vaccinated countries.")
    plt.ylabel("Vaccination Percentage")
    plt.xlabel("Countries")
    plt.xticks(country_num, countries[:10])

    ax = plt.subplot(2, 1, 2)
    plt.ylim(0, 1)
    bottom_countries_figure = plt.bar(
        countries[-10:], percentages[-10:])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title("Bottom 10 vaccinated countries.")
    plt.ylabel("Vaccination Percentage")
    plt.xlabel("Countries")
    plt.xticks(country_num, countries[-10:])

    plt.tight_layout()
    plt.show()

    in_file.close()


def menu():
    print("""\
        
    ______     __                              ____        __           ___                __                     
   / ____/  __/ /_________  ____ ___  ___     / __ \____ _/ /_____ _   /   |  ____  ____ _/ /_  ______  ___  _____
  / __/ | |/_/ __/ ___/ _ \/ __ `__ \/ _ \   / / / / __ `/ __/ __ `/  / /| | / __ \/ __ `/ / / / /_  / / _ \/ ___/
 / /____>  </ /_/ /  /  __/ / / / / /  __/  / /_/ / /_/ / /_/ /_/ /  / ___ |/ / / / /_/ / / /_/ / / /_/  __/ /    
/_____/_/|_|\__/_/   \___/_/ /_/ /_/\___/  /_____/\__,_/\__/\__,_/  /_/  |_/_/ /_/\__,_/_/\__, / /___/\___/_/     
                                                                                         /____/                   

""")

    while True:

        print(
            "\n\n==========================================================================")
        print("Option 1: List last reported week’s number of vaccinated HCW per country.")
        print("Option 2: Display total number of vaccinations per vaccine in a country.")
        print("Option 3: Graph vaccination progress of HCW per week for a country.")
        print("Option 4: Graph top 10 & bottom 10 countries by % of population vaccinated.")
        print("Option 5: Exit.")
        print(
            "==========================================================================\n\n")

        choice = utils.trap(1, 0, 5, 1)

        if choice == 1:
            option1()
        elif choice == 2:
            option2()
        elif choice == 3:
            option3()
        elif choice == 4:
            option4()
        else:
            print("Are you sure you want to exit?")
            bool_choice = utils.trap(2)

            if bool_choice:
                print("Thank you for using the...")
                print("""\
        
    ______     __                              ____        __           ___                __                     
   / ____/  __/ /_________  ____ ___  ___     / __ \____ _/ /_____ _   /   |  ____  ____ _/ /_  ______  ___  _____
  / __/ | |/_/ __/ ___/ _ \/ __ `__ \/ _ \   / / / / __ `/ __/ __ `/  / /| | / __ \/ __ `/ / / / /_  / / _ \/ ___/
 / /____>  </ /_/ /  /  __/ / / / / /  __/  / /_/ / /_/ / /_/ /_/ /  / ___ |/ / / / /_/ / / /_/ / / /_/  __/ /    
/_____/_/|_|\__/_/   \___/_/ /_/ /_/\___/  /_____/\__,_/\__/\__,_/  /_/  |_/_/ /_/\__,_/_/\__, / /___/\___/_/     
                                                                                         /____/                   

""")
                print("{:>80}".format("...have a nice day!"))
                break


menu()
