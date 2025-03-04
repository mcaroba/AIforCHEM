# Copyright (c) 2025 Miguel Caro, Aalto University (miguel.caro@aalto.fi, mcaroba@gmail.com)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


######################################################################
#
# User defined stuff
#
# Data should be a list of entries like [i, j, f(i,j), "exp"|"sim", sigma_reg]
# These are examples (regularization is around OK, fake sample points are out of
# domain so that they don't mess up your fit if you forget to remove them, but
# you should remove them when you add your actual data!):
reg_sim = 0.75; reg_exp = 0.15
data = [ [45, 45, 2.5, "exp", reg_exp], # fake point
         [90, 90, 1.243532135460221, "sim", reg_sim] # fake point
       ]

# Useful information:
#  - the grid is [1:29] x [1:29]
#  - the maximum of the function is 5. and the minimum is 0.
#  - "experimental" measurements will give you a resolution
#    of 1, rounded to the next half integer: if the function's
#    value is 1.2, you'll get 1.5
#  - "simulation" measurements will give you a highly precise
#    number (with many decimals) but with low accuracy: the error
#    will typically be in the [-1:1] interval although the error
#    varies smoothly. This means the simulated function has its
#    minimum possibly displaced from the real function!

# Should we get average and std from the data or do you provide it?
# (if you only have one data point std from data does not make sense;
# in general, you should have a few data points before you start
# training a model)
use_priors = True

# This is so that the string variable ("exp"|"sim") does not mess with numpy
data_array = []
for dat in data:
    data_array.append([dat[0], dat[1], dat[2]])

if use_priors:
    # Spread of values (standard deviation) of the function
    sigma_z = 1.
    # Mean of the function
    mean = 2.5
else:
    sigma_z = np.std(np.array(data_array)[:,2])
    mean = np.mean(np.array(data_array)[:,2])

# Characteristic distance in xy (i-j) plane
sigma_xy = 5.

############################################################



# Do not modify below this line unless you know what you're doing!



############################################################################
#
# Model training
#
z = np.array(data_array)[:,2] - mean
def cov(i1, j1, i2, i3, sigma_z, sigma_xy):
    return sigma_z**2 * np.exp(-0.5*( (i1-i2)**2 + (j1-j2)**2 )/sigma_xy**2 )


# Compute grid properties (where we evaluate the model)
cov_m = np.zeros( [29**2, 29**2] )
for k1 in range(0, 29**2):
    i1 = k1 % 29
    j1 = k1 // 29
    cov_m[k1, k1] = sigma_z**2
    for k2 in range(k1+1, 29**2):
        i2 = k2 % 29
        j2 = k2 // 29
        cov_m[k1, k2] = cov(i1, j1, i2, j2, sigma_z, sigma_xy)
        cov_m[k2, k1] = cov_m[k1, k2]

# Compute data properties
cov_m_data = np.zeros( [len(data), len(data)] )
for k1 in range(0, len(data)):
    i1 = data[k1][0]
    j1 = data[k1][1]
    cov_m_data[k1, k1] = sigma_z**2 + data[k1][4]**2
    for k2 in range(k1+1, len(data)):
        i2 = data[k2][0]
        j2 = data[k2][1]
        cov_m_data[k1, k2] = cov(i1, j1, i2, j2, sigma_z, sigma_xy)
        cov_m_data[k2, k1] = cov_m_data[k1, k2]

cov_m_data_inv = np.linalg.pinv(cov_m_data)

# Compute covariance between training and evaluation points
cov_m_data2 = np.zeros([29**2, len(data)])
alphas = np.dot(cov_m_data_inv, z) # Fitting coefficients (or "weights")
mu_model = np.zeros(29**2)

for k1 in range(0, 29**2):
    i1 = k1 % 29
    j1 = k1 // 29
    covs = np.zeros(len(data))
    for k2 in range(0, len(data)):
        i2 = data[k2][0]
        j2 = data[k2][1]
        covs[k2] = cov(i1, j1, i2, j2, sigma_z, sigma_xy)
    cov_m_data2[k1,:] = covs
#   Best model at the evaluation points
    mu_model[k1] = np.dot(alphas, covs) + mean

# Covariance at the evaluation points (for error estimation)
cov_m_model = cov_m - np.dot(np.dot(cov_m_data2, cov_m_data_inv), np.transpose(cov_m_data2))
############################################################################





############################################################################
#
# File saves
#
# Save current model
f = open("model.dat", "w")
for j in range(0, 29):
    for i in range(0, 29):
        k = i + j*29
        print(i+1, j+1, mu_model[k], np.sqrt(cov_m_model[k,k]), file=f)
    print("", file=f)

f.close()

# Save current data
f = open("data.dat", "w")
for k in range(0, len(data)):
    print(data[k][0], data[k][1], data[k][2], data[k][3], file=f)

f.close()
############################################################################





############################################################################
#
# Plotting
#
# Plot model's current predictions and uncertainty
x = np.zeros([29,29]); y = np.zeros([29,29])
z = np.zeros([29,29]); z_err = np.zeros([29,29])

for j in range(0, 29):
    for i in range(0, 29):
        k = i + j*29
        x[i, j] = i+1
        y[i, j] = j+1
        z[i, j] = mu_model[k]
        z_err[i, j] = np.sqrt(cov_m_model[k,k])

# Matplotlib magic
fig, axs = plt.subplots(1, 2, figsize=(15,5))
cmaplist = [(0.5,0,1,1), (0,0,1,1), (0,1,0,1), (1,1,0,1), (1,0,0,1)]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 5)

# Model
axs[0].set_aspect('equal', 'box')
axs[0].set(xlim=(0,30), ylim=(0,30))
axs[0].set_title("Model")
axs[0].set_xticks(np.arange(0,32,2))
axs[0].set_yticks(np.arange(0,32,2))
psm = axs[0].imshow(np.transpose(z), vmin=0, vmax=5, rasterized=True, cmap=cmap,
                    interpolation="none", origin="lower", extent=(0.5,29.5,0.5,29.5))
fig.colorbar(psm, ax=axs[0], ticks=[0,1,2,3,4,5])
axs[0].plot(x.flatten(), y.flatten(), "+")
x_dat = [data[i][0] for i in range(0, len(data)) if data[i][3] == "exp"]
y_dat = [data[i][1] for i in range(0, len(data)) if data[i][3] == "exp"]
axs[0].plot(x_dat, y_dat, "x", markersize=8, markeredgewidth=3, color="black")
axs[0].plot(x_dat, y_dat, "x", markersize=6, markeredgewidth=1, color="red")
x_dat = [data[i][0] for i in range(0, len(data)) if data[i][3] == "sim"]
y_dat = [data[i][1] for i in range(0, len(data)) if data[i][3] == "sim"]
axs[0].plot(x_dat, y_dat, "x", markersize=8, markeredgewidth=3, color="black")
axs[0].plot(x_dat, y_dat, "x", markersize=6, markeredgewidth=1, color="green")

# Model uncertainty
axs[1].set_aspect('equal', 'box')
axs[1].set(xlim=(0,30), ylim=(0,30))
axs[1].set_title("Model uncertainty")
axs[1].set_xticks(np.arange(0,32,2))
axs[1].set_yticks(np.arange(0,32,2))
psm = axs[1].imshow(np.transpose(z_err), vmin=0, vmax=1.75, rasterized=True, cmap=cmap,
                    interpolation="none", origin="lower", extent=(0.5,29.5,0.5,29.5))
fig.colorbar(psm, ax=axs[1], ticks=[0.,0.25,0.5,0.75,1.,1.25,1.5,1.75])
axs[1].plot(x.flatten(), y.flatten(), "+")
x_dat = [data[i][0] for i in range(0, len(data)) if data[i][3] == "exp"]
y_dat = [data[i][1] for i in range(0, len(data)) if data[i][3] == "exp"]
axs[1].plot(x_dat, y_dat, "x", markersize=8, markeredgewidth=3, color="black")
axs[1].plot(x_dat, y_dat, "x", markersize=6, markeredgewidth=1, color="red")
x_dat = [data[i][0] for i in range(0, len(data)) if data[i][3] == "sim"]
y_dat = [data[i][1] for i in range(0, len(data)) if data[i][3] == "sim"]
axs[1].plot(x_dat, y_dat, "x", markersize=8, markeredgewidth=3, color="black")
axs[1].plot(x_dat, y_dat, "x", markersize=6, markeredgewidth=1, color="green")

plt.show()
############################################################################
