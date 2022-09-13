import ast

import numpy as np

import matplotlib
matplotlib.use("MacOSX")

import matplotlib.pyplot as plt

evolution_fname = "evolution.txt"

L = 10
dim = 150
lims = (-L/2, L/2)
limits = (lims, lims)

with open(evolution_fname) as evolution_fhandle:
    data = []
    for line in evolution_fhandle:
        line = line.strip()
        splitline = line.split()
        for idx in range(len(splitline)):
            elem = splitline[idx]
            elem = ast.literal_eval(elem)
            splitline[idx] = elem[0] + 1j * elem[1]
        data.append(np.array(splitline))

count = 0
for state in data:
    plt.imshow(np.real(np.reshape((state * np.conj(state)), (dim, dim))), extent=[limits[0][0], limits[0][1], limits[1][0], limits[1][1]])
    plt.colorbar()
    plt.title(count)
    plt.show()
    count += 1
