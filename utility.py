import os
import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd

def export_mesh(mesh, label, loc='./'):
    X = mesh.coordinates.dat.data_ro
    triangles = mesh.coordinates.cell_node_map().values_with_halo
    fig, ax = plt.subplots()
    plt.triplot(X[:, 0], X[:, 1], triangles, 'k-', linewidth=0.3)
    ax.set_title(label)
    ax.set_aspect('equal')
    plt.savefig(os.path.join(loc, f"{label}.svg"))
    plt.close()

def export_function(func, label, loc="./"):
    fig, ax = plt.subplots()
    colors = fd.pyplot.tripcolor(func, axes=ax)
    fig.colorbar(colors)
    ax.set_title(label)
    ax.set_aspect('equal')
    plt.savefig(os.path.join(loc, f"{label}.jpg"))
    plt.close()

def export_ufl_function(mesh, func, label, loc="./"):
    u = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    u.interpolate(func)
    export_function(u, label, loc)

### useless
# def sort_mesh(mesh):
#     X = mesh.coordinates.dat.data_ro
#     triangles = mesh.coordinates.cell_node_map().values_with_halo
#     sorted_indices = np.lexsort((X[:,0], X[:,1]))
#     X = X[sorted_indices]
#     idx_map = np.empty(X.shape[0], dtype=int)
#     idx_map[sorted_indices] = np.arange(X.shape[0])
#     triangles = idx_map[triangles]
#     print(X)
#     mesh = fd.Mesh(fd.mesh.plex_from_cell_list(2, triangles, X, comm=fd.COMM_WORLD))