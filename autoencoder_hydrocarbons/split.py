from ase.io import read,write
import numpy as np

# Read in database to be split
db_old = read("db_C9H14.xyz", index=":")


# Split the database into 80% for training and 20% for testing
lst = list(range(0, len(db_old)))
np.random.shuffle(lst) # randomizing, so each time you run this the selection of training/testing structures changes!
db_train = []
db_test = []
for i in range(0, int(len(db_old)*0.8)):
    db_train.append(db_old[lst[i]])

for i in range(int(len(db_old)*0.8), len(db_old)):
    db_test.append(db_old[lst[i]])

write("db_C9H14_train.xyz", db_train)
write("db_C9H14_test.xyz", db_test)
