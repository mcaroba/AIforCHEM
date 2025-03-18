# Copyright (c) 2025 by Miguel A. Caro, Aalto University (miguel.caro@aalto.fi, mcaroba@gmail.com)
import numpy as np
from ase.io import read,write
#from ase.visualize import view
import nglview as nv

# This reads in a database of structures using ASE
db = read("qm9_ch_only_full.xyz", index=":") # https://doi.org/10.5281/zenodo.10925480


# This finds all the formulas in the database
formulas = {}
for atoms in db:
    formula = "C" + str(atoms.symbols.count("C")) + "H" + str(atoms.symbols.count("H"))
    if formula not in formulas:
        formulas[formula] = 1
    else:
        formulas[formula] += 1

# This figures out which formulas have the most isomers in the database and ranks the
# formulas according to number of isomers
ranked_formulas = []
for formula in sorted(formulas, key=formulas.get, reverse=True):
    ranked_formulas.append(formula)
#    print(formula, formulas[formula])

print("The most common formula is " + ranked_formulas[0] + " with " + str(formulas[ranked_formulas[0]]) + " entries")
print("The full list is " + str(list(zip(ranked_formulas, [formulas[ranked_formulas[i]] for i in range(0, len(formulas))]))))

# This accumulates and writes out the structures of the isomers with the most entries in the database for their formula
new_db = []
for atoms in db:
    formula = "C" + str(atoms.symbols.count("C")) + "H" + str(atoms.symbols.count("H"))
    if formula == ranked_formulas[0]:
        new_db.append(atoms)

write("db_" + ranked_formulas[0] + ".xyz", new_db)


# Visualize a randomly picked molecule
nv.show_ase(new_db[np.random.choice(len(new_db))])
