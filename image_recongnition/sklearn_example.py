# This is adapted by Miguel Caro from the sklearn online documentation

# Copyright information provided by the original authors for the sklearn contributions:
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
#
# The modifications by Miguel Caro are Copyright (c) 2025 by Miguel Caro (miguel.caro@aalto.fi)

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA

# Import the digits dataset
digits = datasets.load_digits()

# Plot a sample of all the digits in the database
_, axes0 = plt.subplots(nrows=4, ncols=10, figsize=(10, 5))
axes = []
for irow in range(0, len(axes0)):
    for ax in axes0[irow]:
        axes.append(ax)

for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Label: %i" % label)

# Flatten the images
# We do this so that the 8x8 pixel matrices are turned into 64D vectors
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
print("The shape of the data array is " + str(data.shape))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

# Plot a sample of test set predictions
print("")
print("Predicted vs original label (P:L)")
_, axes0 = plt.subplots(nrows=4, ncols=10, figsize=(10, 5))
axes = []
for irow in range(0, len(axes0)):
    for ax in axes0[irow]:
        axes.append(ax)

for ax, image, prediction, label in zip(axes, X_test, predicted, y_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"{prediction}:{label}")

# Build and plot the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


# Now let's try with different amounts of training data
for train_size in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
    clf = svm.SVC(gamma=0.001) # we need to reinitialize the SVM
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=(1.-train_size), shuffle=True
    )
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix (train_size = " + str(train_size) + ")")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()


# Now let's add some noise
# The value of the data array varies between 0 and 16; we use this to guide
# the choice of the noise level
for max_noise in [0., 0.1, 0.2, 0.5, 1., 2., 5., 10., 20.]:
    data_noisy = np.clip(data + max_noise*2.*(np.random.sample(data.shape) - 0.5), 0, 16)
    clf = svm.SVC(gamma=0.001) # we need to reinitialize the SVM
    X_train, X_test, y_train, y_test = train_test_split(
        data_noisy, digits.target, test_size=0.5, shuffle=True
    )
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix (max_noise = " + str(max_noise) + ")")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()
    _, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 5))
    for ax, image, prediction, label in zip(axes, X_test, predicted, y_test):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{prediction}:{label}")


# And now let's denoise these images using PCA
max_noise = 10.
data_noisy = np.clip(data + max_noise*2.*(np.random.sample(data.shape) - 0.5), 0, 16)
i_list = list(range(len(data)))
np.random.shuffle(i_list)
X_train = data_noisy[i_list[10:]]
X_test = data_noisy[i_list[0:10]]
X_original = data[i_list[0:10]]
N_dim = 10
pca = PCA(n_components=N_dim, random_state=42) # We ask PCA to identify the N_dim dimensions containing the most info
pca.fit(X_train)
# Try also kernel PCA
N_dim = 100
kernel_pca = KernelPCA(
    n_components=N_dim,
    kernel="rbf",
    gamma=1e-3,
    fit_inverse_transform=True,
    alpha=5e-3,
    random_state=42,
)
_ = kernel_pca.fit(X_train)

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test)) # We project the N_dim representation back to the 64D space
X_reconstructed_kpca = kernel_pca.inverse_transform(kernel_pca.transform(X_test))

_, axes = plt.subplots(nrows=4, ncols=10, figsize=(10, 5))
for ax, image in zip(axes[0], X_original):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Original")
for ax, image in zip(axes[1], X_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Noisy")
for ax, image in zip(axes[2], X_reconstructed_pca):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"DN-PCA")
for ax, image in zip(axes[3], X_reconstructed_kpca):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"DN-kPCA")
