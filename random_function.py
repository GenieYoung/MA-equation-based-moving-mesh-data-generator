import random
import numpy as np
import firedrake as fd
from scipy.interpolate import griddata

seed = 42
random.seed(seed)
np.random.seed(seed)

def random_ufl_func(max_dist=6):
    n_dist = random.randint(1, max_dist)
    x_μ_dict = [random.uniform(0,1) for i in range(n_dist)]
    y_μ_dict = [random.uniform(0,1) for i in range(n_dist)]
    x_σ_dict = [max(round(random.gauss(0.25,0.16),3), 0.125) for i in range(n_dist)]
    y_σ_dict = [max(round(random.gauss(0.25,0.16),3), 0.125) for i in range(n_dist)]
    z_dict = [random.uniform(5,20) for i in range(n_dist)]
    def func(mesh):
        x, y = fd.SpatialCoordinate(mesh)
        #return fd.ufl.tanh(-30*(y-0.5-0.25*fd.ufl.sin(2*fd.ufl.pi*x)))
        source = 0
        for i in range(n_dist):
            source += z_dict[i] * fd.exp(
                -1 * ((((x - x_μ_dict[i]) ** 2) / (x_σ_dict[i]**2)) + (((y - y_μ_dict[i]) ** 2) / (y_σ_dict[i]**2))))
        return source
    return func

def random_func(max_dist=6, degree=1):
    ufl_func = random_ufl_func(max_dist)
    def func(mesh):
        V = fd.FunctionSpace(mesh, "CG", degree)
        F = fd.Function(V)
        F.interpolate(ufl_func(mesh))
        #F.interpolate(fd.ufl.sin(2*x) + fd.ufl.sin(2*y))
        return F
    return func

def random_func_gp(degree=1):
    N = 101
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x, indexing='ij')

    kx = 2 * np.pi * np.fft.fftfreq(N, d=x[1]-x[0])
    KY, KX = np.meshgrid(kx, kx, indexing='ij')
    l = 0.2
    sigma = 1
    S = sigma**2 * (2*np.pi*l**2) * np.exp(-0.5 * l**2 * (KX**2 + KY**2))

    rng = np.random.default_rng()
    white = rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))
    Z_fft = np.fft.ifft2(white * np.sqrt(S), norm='ortho').real

    coords_reg = np.stack([X.ravel(), Y.ravel()], -1)
    vals_reg   = Z_fft.ravel() * 1000

    # M = fd.UnitSquareMesh(100, 100)
    # X = M.coordinates.dat.data_ro
    # V = fd.FunctionSpace(M, "CG", 1)
    # initial_F = fd.Function(V)
    # initial_F.vector()[:] = griddata(coords_reg, vals_reg,
    #                             M.coordinates.dat.data_ro,
    #                             method='linear', fill_value=0.0)

    def func(mesh):
        V = fd.FunctionSpace(mesh, "CG", degree)
        F = fd.Function(V)
        F.dat.data[:] = griddata(coords_reg, vals_reg,
                                fd.mg.utils.physical_node_locations(V).dat.data[:],  # get coordinates of all DOF nodes
                                method='linear', fill_value=0.0)

        # vom = fd.VertexOnlyMesh(M, fd.mg.utils.physical_node_locations(V).dat.data[:],redundant=False)
        # vom_space = fd.FunctionSpace(vom, "DG", 0)
        # target_vom_f = fd.Function(vom_space)
        # target_vom_f.interpolate(initial_F)
        # F.dat.data[:] = target_vom_f.dat.data

        return F
    return func

def random_func_pcg(degree=1):
    M = fd.UnitSquareMesh(100, 100)
    V = fd.FunctionSpace(M, "CG", degree)
    pcg = np.random.SFC64(seed)
    rg = fd.RandomGenerator(pcg)
    initial_F = rg.beta(V, 1, 2)
    def func(mesh):
        V = fd.FunctionSpace(mesh, "CG", 2)
        F = fd.Function(V)
        F.interpolate(initial_F)
        return F
    return func