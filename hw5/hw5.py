import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    #filename = sys.argv[1]
    filename = "hw5.csv"
    #filename = "toy.csv"

    year = list()
    days = list()
    with open(filename) as csv:
        for line in csv:
            entries = line.split(",")
            if entries[0] == "year":
                continue
            year.append(int(entries[0]))
            days.append(int(entries[1][:-1]))


    # Question 2
    plt.plot(year, days)
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.title("Lake Mendota Frozen Days")
    plt.savefig("plot.jpg")

    # Question 3
    print("Q3a:")
    num_x_variables = 1
    X = np.ones((len(year), 1 + num_x_variables))
    for i_year in range(len(year)):
        x_i = np.array([int(1), int(year[i_year])])
        X[i_year, :] = x_i
    print(X)

    print("Q3b:")
    Y = np.ones(len(days))
    for i_day in range(len(days)):
        Y[i_day] = days[i_day]
    print(Y)

    print("Q3c:")
    Z = np.dot(X.T, X)
    print(Z)

    print("Q3d:")
    I = np.linalg.inv(Z)
    print(I)

    print("Q3e:")
    PI = np.dot(I, X.T)
    print(PI)

    print("Q3f:")
    hat_beta = np.dot(PI, Y)
    print(hat_beta)

    # Question 4
    y_test = hat_beta[0] + hat_beta[1]*2021
    print("Q4: " + str(y_test))

    # Question 5
    if hat_beta[1] > 0:
        print("Q5a: >")
        print("Q5b: This means that for every additional year, the expected number"
              " of days the lake is frozen for will go up")
    elif hat_beta[1] < 0:
        print("Q5a: <")
        print("Q5b: This means that for every additional year, the expected number"
              " of days the lake is frozen for will go down")
    else:
        print("Q5a: =")
        print("Q5b: This means that for every additional year, the expected number"
              " of days the lake is frozen for will stay the same")


    # Question 6
    x_star = -1*(hat_beta[0]/hat_beta[1])
    print("Q6a: " + str(x_star))
    print("Q6b: Based on trends in the data, the estimation of lakes no longer freezing by " + str(x_star)
          + " is plausible if we assume relationship we are looking at truly is linear. However, given "
            "outside knowledge of climate change, this assumption is sketchy. If the Earth heats up too "
            "much, it'll cause positive feedback loops that will make the temperature increase exponentially "
            " and non-linearly. If this ends up being the case, the estimated year lakes stop freezing by "
            "would be much earlier than " + str(x_star) + ".")

