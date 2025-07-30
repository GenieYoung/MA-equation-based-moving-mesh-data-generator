import firedrake as fd

def monitor_func(u_func):
    def monitor(mesh):
        V = fd.FunctionSpace(mesh, "CG", 2)
        H = fd.grad(fd.grad(u_func(mesh)))
        Hnorm = fd.Function(V, name="Hnorm")
        Hnorm.interpolate(fd.sqrt(fd.inner(H, H)))
        Hnorm_max = Hnorm.dat.data.max()
        m = 1 + 10 * Hnorm / Hnorm_max
        return m
    return monitor