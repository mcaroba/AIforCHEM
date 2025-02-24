# Copyright (c) 2025 Miguel A. Caro, Aalto University (miguel.caro@aalto.fi)

#############################################################################################
#
# We work with an example where we try to predict the temperature at a given location as a
# function of the day of the year
#
#############################################################################################


#############################################################################################
#
# Imports
#
import numpy as np
#
#############################################################################################


#############################################################################################
#
# Most commonly modified user parameters
#
# Number of example functions we're going to generate
N_func = 10
#
# Helper dictionary that gives day number in middle of requested month
day = {"Jan": 15.5, "Feb": 45., "Mar": 74.5, "Apr": 105., "May": 135.5, "Jun": 166.,
       "Jul": 196.5, "Aug": 227.5, "Sep": 258., "Oct": 288.5, "Nov": 319., "Dec": 349.5}
#
# Observations of time, temperature in [day, K] units
#
#data = []
#data = [[day["Jan"], -3.1],
#        [day["Jul"], 18.1],
#        [day["Oct"],  6.6]]
data = np.loadtxt("temp_helsinki_series.dat")
#
#############################################################################################


#############################################################################################
#
# Let's provide some priors (assumptions before we observe any data)
#
# Average temperature
if len(data) == 0:
    T_av = 10. # degrees Celsius
else:
    T_av = np.mean(np.array(data)[:,1])

# How much, on average, does the average temperature change over the course of a day?
dTdt = 5./30. # degrees Celsius/day
#
# What's the expected spread of temperatures throughout the year?
# E.g., 1/2 of T_July - T_January
sigma_T = 12.5 # degrees Celsius
#
# For Gaussian noise regularization
sigma_reg = 0.
#sigma_reg = 0.1*sigma_T
#
#############################################################################################

#############################################################################################
#
# What do these functions look like before we observe any data?
#
# Number of "random variables" (number of points in which the time axis is discretized)
N_rand = 365 # One for each day of the year but we could do something else
#
# We define the covariance function. Here we use a smooth Gaussian kernel so that our final functions
# are also smooth (if t1 and t2 are nearby, the covariance is close to T_av**2)
sigma_t = sigma_T/dTdt
def cov(t1, t2, sigma_t, sigma_T):
    return sigma_T**2 * np.exp(-0.5*(t1-t2)**2/sigma_t**2)
#    return 0. # uncomment this if you want independent variables

# The distance in time between random variable locations
dt = 365./N_rand
#
#############################################################################################




# Average function
# If we don't have data, our prior is given by the average temperature
if len(data) == 0:
    mu = np.zeros(N_rand) + T_av
# If we have data, we compute the mean vector from the actual observations, assuming
# T_* = \sum_s \alpha_s cov(*,s); the fitting task is to derive the \alpha_s from the data
else:
    Ts = np.array(data)[:,1] # All the temperatures in the dataset
    ts = np.array(data)[:,0] # All the times in the dataset
    # Construct the covariance matrix of the data
    cov_m_data = np.zeros([len(data), len(data)])
    cov_m_data2 = np.zeros([N_rand, len(data)]) # this we save for later
    for i in range(0, len(data)):
        t1 = ts[i]
        cov_m_data[i,i] = sigma_T**2 + sigma_reg**2
        for j in range(i+1, len(data)):
            t2 = ts[j]
            cov_m_data[i,j] = cov(t1, t2, sigma_t, sigma_T)
            cov_m_data[j,i] = cov_m_data[i,j]
    cov_m_data_inv = np.linalg.pinv(cov_m_data)
    alphas = np.dot(cov_m_data_inv, Ts) # Fitting coefficients (or "weights")
    mu = np.zeros(N_rand) # Just to initialize the mu vector
    for i in range(0, N_rand):
        t1 = i*dt
        covs = cov(t1, ts, sigma_t, sigma_T) # vector of covariances between sampling point t1 and data locations ts
        cov_m_data2[i,:] = covs # this we save for later
        mu[i] = np.dot(alphas, covs)


# Covariance between the random variables in the absence of data
cov_m = np.zeros([N_rand, N_rand])
for i in range(0, N_rand):
    t1 = i*dt
    cov_m[i,i] = sigma_T**2
    for j in range(i+1, N_rand):
        t2 = j*dt
        cov_m[i,j] = cov(t1, t2, sigma_t, sigma_T)
        cov_m[j,i] = cov_m[i,j]

# If we have data, we need to condition the functions to match the data by modifying the covariance:
if len(data) > 0:
    cov_m = cov_m - np.dot(np.dot(cov_m_data2, cov_m_data_inv), np.transpose(cov_m_data2))



# Draw N_func random samples (functions) from our multivariate normal distribution compatible
# with the previously defined mean and covariance matrix. Each sample is a vector of dimension
# N_rand
samples = np.random.multivariate_normal(mu, cov_m, N_func)
#
# Save them for plotting
f = open("functions.dat", "w")
for sample in samples:
    for i in range(0, len(sample)):
        print(i*dt, sample[i], file=f)
    print("", file=f)

f.close()

f = open("functions_av.dat", "w")
samples_av = np.mean(samples, axis=0)
samples_std = np.std(samples, axis=0)
for i in range(0, len(samples_av)):
    print(i*dt, samples_av[i], samples_std[i], file=f)
print("", file=f)

f.close()

f = open("functions_multivariate.dat", "w")
for i in range(0, len(mu)):
    print(i*dt, mu[i], np.sqrt(cov_m[i,i]), file=f)
print("", file=f)

f.close()

f = open("data.dat", "w")
for i in range(0, len(data)):
    print(data[i][0], data[i][1], file=f)
print("", file=f)

f.close()
