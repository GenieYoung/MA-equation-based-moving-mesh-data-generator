from utility import *
from random_function import *
from monitor_function import *
from movement import MongeAmpereMover

# build mesh
n_x = 100
n_y = 100
mesh = fd.UnitSquareMesh(n_x, n_y)

n_samples = 1

import time
t0 = time.time()

for i in range(n_samples):
    print(f"================ Sample {i+1}/{n_samples} ==================")

    F = random_func(degree=1)
    export_function(F(mesh), f"u_{i}", "u_exact/")
    # F = random_ufl_func()
    # export_ufl_function(mesh, F(mesh), f"u_{i}", "u_exact/")

    export_ufl_function(mesh, monitor_func2(F)(mesh), "monitor")

    mover = MongeAmpereMover(mesh, monitor_func2(F), method="relaxation", rtol=1e-08)
    mover.move()
    export_mesh(mover.mesh, f"mesh_{i}", "adaptive_mesh/")

print(f"Time taken: {time.time() - t0:.2f} seconds")
