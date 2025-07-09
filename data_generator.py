import random
import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from movement import MongeAmpereMover

random.seed(43)

def export_mesh(mesh, label, loc="./"):
    # fig,ax = plt.subplots()
    # fd.pyplot.triplot(mesh, axes=ax)
    # ax.set_title(label)
    # ax.set_aspect('equal')
    # plt.savefig(f"{loc}{label}.jpg")
    X = mesh.coordinates.dat.data_ro
    triangles = mesh.coordinates.cell_node_map().values_with_halo
    fig, ax = plt.subplots()
    plt.triplot(X[:, 0], X[:, 1], triangles, 'k-', linewidth=0.3)
    ax.set_title(label)
    ax.set_aspect('equal')
    plt.savefig(f"{loc}{label}.svg")
    plt.close()
    
def export_function(func, label, loc="./"):
    fig, ax = plt.subplots()
    colors = fd.pyplot.tripcolor(func, axes=ax)
    fig.colorbar(colors)
    ax.set_title(label)
    ax.set_aspect('equal')
    plt.savefig(f"{loc}{label}.jpg")
    plt.close()
    
def export_ufl_function(mesh, func, label, loc="./"):
    fig, ax = plt.subplots()
    u = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    u.interpolate(func)
    colors = fd.pyplot.tripcolor(u, axes=ax)
    fig.colorbar(colors)
    ax.set_title(label)
    ax.set_aspect('equal')
    plt.savefig(f"{loc}{label}.jpg")
    plt.close()
    
def get_values(mesh, func):
    V = fd.FunctionSpace(mesh, "CG", 1)
    F = fd.Function(V)
    F.interpolate(func)
    return F.dat.data
    
def rand_u_exact(max_dist=6):
    n_dist = random.randint(1, max_dist)
    x_μ_dict = [random.uniform(0.2,0.8) for i in range(n_dist)]
    y_μ_dict = [random.uniform(0.2,0.8) for i in range(n_dist)]
    x_σ_dict = [max(round(random.gauss(0.25,0.16),3), 0.125) for i in range(n_dist)]
    y_σ_dict = [max(round(random.gauss(0.25,0.16),3), 0.125) for i in range(n_dist)]
    z_dict = [random.uniform(5,20) for i in range(n_dist)]
    def u(mesh):
        x, y = fd.SpatialCoordinate(mesh)
        #return fd.ufl.tanh(-30*(y-0.5-0.25*fd.ufl.sin(2*fd.ufl.pi*x)))
        source = 0
        for i in range(n_dist):
            source += z_dict[i] * fd.exp(
                -1 * ((((x - x_μ_dict[i]) ** 2) / (x_σ_dict[i]**2)) + (((y - y_μ_dict[i]) ** 2) / (y_σ_dict[i]**2))))
        return source
    return u

def monitor_exact(u_exact):
    def monitor_func(mesh):
        if False:
            return fd.ufl.sqrt(1+0.1*fd.ufl.inner(fd.grad(u_exact(mesh)), fd.grad(u_exact(mesh))))
        else:
            V = fd.FunctionSpace(mesh, "CG", 1)
            Hnorm = fd.Function(V, name="Hnorm")
            H = fd.grad(fd.grad(u_exact(mesh)))
            Hnorm.interpolate(fd.sqrt(fd.inner(H, H)))
            Hnorm_max = Hnorm.dat.data.max()
            m = 1 + 10 * Hnorm / Hnorm_max
            return m
    return monitor_func

def move_mesh(initial_mesh, u_exact):
    rtol = 1.0e-08
    #export_ufl_function(initial_mesh, monitor_exact(u_exact)(initial_mesh), "test1_monitor_exact")
    mover = MongeAmpereMover(initial_mesh, monitor_exact(u_exact), method="relaxation", rtol=rtol)
    mover.move()
    return mover
    
n = 100
mesh = fd.UnitSquareMesh(n, n)  # initial mesh
#export_mesh(mesh, "test1_initial_mesh")

# u_exact = rand_u_exact()
# export_ufl_function(mesh, u_exact(mesh), "test1_u_exact")

# res = move_mesh(mesh, u_exact)
# export_mesh(res.mesh, "test1_moved_mesh")
# export_function(res.phi, "test1_moved_solution")

n_samples = 100
branch_input_train = np.zeros((n_samples, (n+1)*(n+1)))
trunk_input_train = np.array(mesh.coordinates.dat.data)
output_train = np.zeros((n_samples,(n+1)*(n+1),2))
triangles = np.array(mesh.coordinates.cell_node_map().values_with_halo)

import time
t0 = time.time()

for i in range(n_samples):
    print(f"================ Sample {i+1}/{n_samples} ==================")
    u_exact = rand_u_exact()
    export_ufl_function(mesh, u_exact(mesh), f"u_exact_{100+i}", "u_exact/")
    branch_input_train[i,:] = get_values(mesh, u_exact(mesh))
        
    res = move_mesh(mesh, u_exact)
    output_train[i,:,:] = res.mesh.coordinates.dat.data
    export_mesh(res.mesh, f"adaptive_mesh_{100+i}", "mesh/")
    
print(f"Time taken: {time.time() - t0:.2f} seconds")
    
np.savez("MA_data.npz", X0=branch_input_train, X1=trunk_input_train, Y=output_train, T=triangles)



