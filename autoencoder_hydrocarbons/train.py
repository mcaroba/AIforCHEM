# Copyright (c) 2025 by Miguel A. Caro, Aalto University (miguel.caro@aalto.fi, mcaroba@gmail.com)
import numpy as np
from ase.io import read,write
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from ase import Atoms
#from ase.visualize import view
import nglview as nv


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
#db = read("qm9_ch_only_full.xyz", index=":")
#db = read("db_C9H14.xyz", index=":")
db = read("db_C9H14_rattled.xyz", index=":")
#db = read("db_C9H14_train.xyz", index=":")


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


# Sanity check that we can embed the structure back into Cartesian space
# First we view the regular representation (here I pick the first structure in the database)
nv.show_ase(db[0])
# Now we make Coulomb, then flatten, then unflatten, then embed, then plot
cm, dist = coulomb_matrix(db[0], use_charge)
v = flatten_upper_triangle(cm)
cm, dist = unflatten_upper_triangle(v, use_charge, 9, 14)
mds = MDS(n_components=3, dissimilarity="precomputed", n_init=100)
xyz = mds.fit_transform(dist)
at = Atoms("C9H14", positions = xyz)
nv.show_ase(at)


# This is the Relu activation function that takes a scalar as argument
def relu(x):
    return np.max([0., x])


# This builds the descriptors for all the isomers of the chosen formula
desc = []
for atoms in db:
    cm, dist = coulomb_matrix(atoms, use_charge)
    desc.append(flatten_upper_triangle(cm))


# This builds the neural network object and trains it according to an autoencoder architecture
hidden_layers = []
for i in range(0, len(encoder_hidden_layers)):
    hidden_layers.append(encoder_hidden_layers[i])

for i in range(len(encoder_hidden_layers)-2, -1, -1):
    hidden_layers.append(encoder_hidden_layers[i])

print("Your chosen autoencoder architecture is " + str(hidden_layers))

print("Training neural network...")
ae = MLPRegressor(solver='adam', hidden_layer_sizes=hidden_layers, max_iter=2000, random_state=1, alpha=alpha)
ae.fit(desc, desc)

weights = ae.coefs_
bias = ae.intercepts_
print("   ... done.")


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


# Calculate RMSE within the training database (not too bad because we're looking at how well
# we can compress, rather than how well we can predict other properties)
RMSE_train = 0.
for i in range(0, len(desc)):
    v = autoencode(weights, bias, desc[i])
    RMSE_train += np.dot(desc[i]-v, desc[i]-v)

RMSE_train = np.sqrt(RMSE_train/len(desc))
print("RMSE_train = " + str(RMSE_train))


# Now let's do this on a test database
db_test = read("db_C9H14.xyz", index=":")
#db_test = read("db_C9H14_test.xyz", index=":")
desc = []
for atoms in db_test:
    cm, dist = coulomb_matrix(atoms, use_charge)
    desc.append(flatten_upper_triangle(cm))

RMSE_test = 0.
for i in range(0, len(desc)):
    v = autoencode(weights, bias, desc[i])
    RMSE_test += np.dot(desc[i]-v, desc[i]-v)

RMSE_test = np.sqrt(RMSE_test/len(desc))
print("RMSE_test = " + str(RMSE_test))


# Create a library of compressed descriptors
compressed_lib = []
for i in range(0, len(desc)):
    v = encode(weights, bias, desc[i])
    compressed_lib.append(v)


# Some visual checks of the quality of the embedding
i_rand = np.random.choice(len(desc))
m0, dist0 = unflatten_upper_triangle(desc[i_rand], use_charge, 9, 14) # Coulomb matrix of first structure
v = autoencode(weights, bias, desc[i_rand])
m, dist = unflatten_upper_triangle(v, use_charge, 9, 14) # Autoencoded Coulomb matrix of first structure

# Get Cartesian coordinates from Coulomg matrix
mds = MDS(n_components=3, dissimilarity="precomputed", n_init=100)
xyz0 = mds.fit_transform(dist0) # Original CM
xyz = mds.fit_transform(dist) # Autoencoded CM

atoms0 = Atoms("C9H14", positions=xyz0)
atoms = Atoms("C9H14", positions=xyz)

#view([db_test[0], atoms0, atoms])
nv.show_ase(atoms0)

nv.show_ase(atoms)


# Let's cluster the data in the latent space with k-means
kmeans = KMeans(n_clusters=5, random_state=0, n_init=100).fit(compressed_lib)
labels = kmeans.labels_ # List with the cluster number each molecule belongs to
centers = kmeans.cluster_centers_ # The coordinates in latent space of the centroids
# Turn these centers into molecules (the most representative)
at_centers = []
for i in range(0, len(centers)):
    v = decode(weights, bias, centers[i])
    m, dist = unflatten_upper_triangle(v, use_charge, 9, 14)
    xyz = mds.fit_transform(dist)
    atoms = Atoms("C9H14", positions=xyz)
    at_centers.append(atoms)

nv.show_ase(at_centers[0])
