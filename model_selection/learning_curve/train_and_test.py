# Copyright (c) 2025 by Miguel A. Caro, Aalto University (miguel.caro@aalto.fi, mcaroba@gmail.com)
import numpy as np
from ase.io import read,write
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from ase import Atoms
import matplotlib as mpl
import matplotlib.pyplot as plt


####################################################################################
#
# User-editable fields
#
# Autoencoder structure; we define the encoder's architecture and the decoder uses
# a symmetric architecture which we do not define explicitly
# I.e., we set encoder_hidden_layers = [nn_1, nn_2, ..., nn_NE] (nn = number of neurons;
# NE = number of encoder hidden layers; i.e., nn_1 are the number of neurons in hidden layer 1)
# and the autoencoder architecture will be [nn_1, nn_2, ..., nn_NE, ..., nn_2, nn_1]
# It's important to note that nn_NE defines the number of dimensions in the encoding
# space, and it should be lower than the original number of dimensions of the feature
# vector. In this example, without any further modifications, our input vectors have
# dimensions 253, but this would depend on the number of atoms. When encoding, there
# is a balance you need to strike: high dimensions nn_NE will lead to less lossy
# compression (you retain more information), but will make the optimization in the
# embedding space more challenging. Lower nn_NE will make this optimization easier
# but will retain less information after compression.
encoder_hidden_layers = [400, 200, 50]
use_charge = False # Should we use the nuclear charges in the definition of the Coulomb matrix?
alpha = 0.001 # Strength of regularization term; default is 0.0001
####################################################################################


# Don't change things below unless you know what you're doing



# This reads in a database of structures using ASE
db = read("db_C9H14.xyz", index=":")
#db = read("db_C9H14_rattled.xyz", index=":")


# This function computes the Coulomb matrix for an ASE's Atoms object
def coulomb_matrix(atoms, use_charge, sort_by_species=True):
    cm = np.zeros([len(atoms), len(atoms)])
    dist = np.zeros([len(atoms), len(atoms)])
    if use_charge:
        Z = atoms.numbers
    else:
        Z = np.zeros(len(atoms)) + 1.
    for i in range(0, len(atoms)):
        cm[i,i] = 0.5 * Z[i]**2.4
        dist[i,i] = 0.
        for j in range(i+1, len(atoms)):
            dist[i, j] = atoms.get_distance(i,j)
            dist[j, i] = dist[i, j]
            cm[i, j] = Z[i]*Z[j] / dist[i, j]
            cm[j, i] = cm[i, j]
    if sort_by_species:
        cm_sort_rows = cm[np.argsort(atoms.numbers)[::-1], :]
        cm_sorted = cm_sort_rows[:, np.argsort(atoms.numbers)[::-1]]
        return cm_sorted, dist
    else:
        return cm, dist


# This function retains the upper triangle (without the diagonal elements) of the
# Coulmb matrix and expresses it as a vector
def flatten_upper_triangle(m):
    flat = []
    assert m.shape[0] == m.shape[1]
    for i in range(0, m.shape[0]):
        for j in range(i+1, m.shape[1]):
            flat.append(m[i,j])
    return np.array(flat)

def unflatten_upper_triangle(flat, use_charge, nC, nH):
    N = int(1+np.sqrt(1+8*len(flat)))//2
    m = np.zeros([N,N])
    dist = np.zeros([N,N])
    k = 0
    for i in range(0, N):
        if i < nC:
            if use_charge:
                Zi = 6.
            else:
                Zi = 1.
        else:
            Zi = 1.
        m[i,i] = 0.5*Zi**2.4
        for j in range(i+1, N):
            if j < nC:
                if use_charge:
                    Zj = 6.
                else:
                    Zj = 1.
            else:
                Zj = 1.
            dist[i, j] = np.max([0., Zi*Zj/flat[k]])
            dist[j, i] = np.max([0., Zi*Zj/flat[k]])
            m[i,j] = flat[k]
            m[j,i] = flat[k]
            k += 1
    return m, dist


# This is the Relu activation function that takes a scalar as argument
def relu(x):
    return np.max([0., x])


# This builds the descriptors for all the isomers of the chosen formula
desc = []
for atoms in db:
    cm, dist = coulomb_matrix(atoms, use_charge)
    desc.append(flatten_upper_triangle(cm))


# Select 20% for testing and the rest for training
random_list = list(range(0,len(desc)))
np.random.shuffle(random_list)
desc_test = [desc[i] for i in random_list[0:int(0.2*len(desc))]]
desc_train = [desc[i] for i in random_list[int(0.2*len(desc)):len(desc)]]


# Define the function to autoencode
def autoencode(weights, bias, this_desc):
    v = this_desc.copy()
    for i in range(0, len(bias)):
        v_new = np.dot(np.transpose(weights[i]), v) + bias[i]
        if i < len(bias)-1:
            v = np.array([relu(a) for a in v_new])
        else:
            v = v_new
    return v


# Define the function to encode (assumes the NN has symmetric layer architecture)
def encode(weights, bias, this_desc):
    v = this_desc.copy()
    for i in range(0, len(bias)//2):
        v_new = np.dot(np.transpose(weights[i]), v) + bias[i]
        if i < len(bias)//2-1:
            v = np.array([relu(a) for a in v_new])
        else:
            v = v_new
    return v


# Define the function to decode (assumes the NN has symmetric layer architecture)
def decode(weights, bias, this_compressed_desc):
    v_new = this_compressed_desc.copy()
    v = np.array([relu(a) for a in v_new])
    for i in range(len(bias)//2, len(bias)):
        v_new = np.dot(np.transpose(weights[i]), v) + bias[i]
        if i < len(bias)-1:
            v = np.array([relu(a) for a in v_new])
        else:
            v = v_new
    return v


# This builds the neural network object
hidden_layers = []
for i in range(0, len(encoder_hidden_layers)):
    hidden_layers.append(encoder_hidden_layers[i])

for i in range(len(encoder_hidden_layers)-2, -1, -1):
    hidden_layers.append(encoder_hidden_layers[i])

print("# Your chosen autoencoder architecture is " + str(hidden_layers))

print("# Training neural network...")
ae = MLPRegressor(solver='adam', hidden_layer_sizes=hidden_layers, max_iter=2000, random_state=1, alpha=alpha)


# This trains the autoencoder with an increasing fraction of the training data and tests it
# on the test data
results = []
for f in np.arange(0.05, 1.+1.e-10, 0.05):
    ae.fit(desc_train[0:int(f*len(desc_train))], desc_train[0:int(f*len(desc_train))])
    weights = ae.coefs_
    bias = ae.intercepts_
    # Testing
    RMSE_test = 0.
    for i in range(0, len(desc_test)):
        v = autoencode(weights, bias, desc_test[i])
        RMSE_test += np.dot(desc_test[i]-v, desc_test[i]-v)
    RMSE_test = np.sqrt(RMSE_test/len(desc_test))
    print(int(f*len(desc_train)), RMSE_test)
    results.append([int(f*len(desc_train)), RMSE_test])

results = np.array(results)

print("#   ... training done.")


# This plots the results
fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].axhline(y=1.2930, color="black", linestyle="--") # This is what I got with 80% of the regular database
ax[0].axhline(y=0.4269, color="black", linestyle="--") # This is what I got with 80% of the x100 rattled database
ax[0].plot(results[:,0], results[:,1], linewidth=2, linestyle="-")
ax[0].set(xlabel="Number of training points")
ax[0].set(ylabel="Test-set RMSE")

ax[1].axhline(y=1.2930, color="black", linestyle="--") # This is what I got with 80% of the regular database
ax[1].axhline(y=0.4269, color="black", linestyle="--") # This is what I got with 80% of the x100 rattled database
ax[1].plot(np.log10(results[:,0]), results[:,1], linewidth=2, linestyle="-")
ax[1].set(xlabel="log10 of the number of training points")
ax[1].set(ylabel="Test-set RMSE")

plt.show()
