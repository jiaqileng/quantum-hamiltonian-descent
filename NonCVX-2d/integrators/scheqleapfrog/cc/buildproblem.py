import numpy as np

import scipy.sparse as sps
import scipy.stats as spstat

import matplotlib
matplotlib.use("MacOSX")

import matplotlib.pyplot as plt

"""
dims: list or tuple of numbers indicating how many steps along each axis x, y
limits: list of tuples indicating the low and high values of each axis x, y
means: tuple of means to center the 2d gaussian at
sigmas: tuple of sigmas for each dimension
"""
def build_2d_gaussian(output_fname, dims, limits, mu, cov):
    x_low, x_high = limits[0]
    x_dom = np.linspace(x_low, x_high, dims[0])

    y_low, y_high = limits[1]
    y_dom = np.linspace(y_low, y_high, dims[1])

    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    pdf = np.empty(dims, dtype='complex')

    for x_idx in range(dims[0]):
        for y_idx in range(dims[1]):
            point = np.transpose(np.array((x_dom[x_idx], y_dom[y_idx])))
            pdf[x_idx, y_idx] = spstat.multivariate_normal(mean=mu, cov=cov).pdf([x_dom[x_idx], y_dom[y_idx]])

    # plt.imshow(pdf.real, extent=[limits[0][0], limits[0][1], limits[1][0], limits[1][1]])
    # plt.colorbar()
    # plt.show()

    with open(output_fname, "w+") as ofile:
        for elem in pdf.flat:
            # print(elem)
            ofile.write("{0.real:.20f} {0.imag:.20f}\n".format(elem))


def build_2d_laplacian(output_fname, dims):
    if dims[0] != dims[1]:
        raise ValueError("dimensions must be square due to implementation")

    n = dims[0]
    A = sps.eye(n, k=-1) - 4*sps.identity(n) + sps.eye(n, k=1)
    B = sps.eye(n, k=-1) + sps.eye(n, k=1)

    lap2d = (sps.kron(sps.identity(n), A) + sps.kron(B, sps.identity(n))).tocoo()
    lap2d.eliminate_zeros()

    with open(output_fname, "w+") as ofile:
        for elem_idx in range(lap2d.nnz):
            ofile.write("{0} {1} {2}\n".format(lap2d.row[elem_idx],
                                               lap2d.col[elem_idx],
                                               lap2d.data[elem_idx]))


dim = 150
L = 10
lims = (-L/2, L/2)
mu = np.array([[0],
               [0]])
cov = np.array([[1, 0],
                [0, 1]])

build_2d_laplacian("lap2d.txt",
                   (dim, dim))


build_2d_gaussian("gaussian.txt",
                  (dim, dim),
                  (lims, lims),
                  (0, 0),
                  cov)
