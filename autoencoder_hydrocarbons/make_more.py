from ase.io import read,write
import numpy as np

db_old = read("db_C9H14.xyz", index=":")

max_rattle = 0.05

db_new = []
for atoms in db_old:
    for i in range(0, 10):
        atoms_cp = atoms.copy()
        atoms_cp.positions += 2.*max_rattle*(np.random.sample([len(atoms), 3])-0.5)
        db_new.append(atoms_cp)

write("db_C9H14_rattled.xyz", db_new)
