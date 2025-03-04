import numpy as np

################################################
#
# You can modify here
#
mode = "exp"
#mode = "sim"
i = 1
j = 1
################################################





# Do not modify below this line and do not look at the files,
# it's cheating!





exp = np.loadtxt("function.dat")
sim = np.loadtxt("noise.dat")
sim[:,2] += exp[:,2]

try:
    f = open("measurements.dat", "r")
    lines = f.readlines()
    f.close()
except:
    lines = []

f = open("measurements.dat", "a")

budget = 60.
for line in lines:
    this_mode = line.split()[3]
    if this_mode == "exp":
        budget -= 5.
    elif this_mode == "sim":
        budget -= 1.

if mode == "exp":
    cost = 5.
elif mode == "sim":
    cost = 1.

if budget - cost < 0.:
    print("You don't have enough money left to do a " + mode + "!")
else:
    if mode == "exp":
        for data in exp:
            if i == data[0] and j == data[1]:
                print(i, j, np.floor(data[2])+0.5, "exp", file=f)
                print("This is the new data (i, j, f(i,j), mode):")
                print(i, j, np.floor(data[2])+0.5, "exp")
    elif mode == "sim":
        for data in sim:
            if i == data[0] and j == data[1]:
                print(i, j, data[2], "sim", file=f)
                print("This is the new data (i, j, f(i,j), mode):")
                print(i, j, data[2], "sim")
    print("Your remaining budget is ", budget-cost, "Alvars")

f.close()
