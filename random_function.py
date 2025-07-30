import random
import firedrake as fd

random.seed(42)

def random_func(max_dist=6):
    n_dist = random.randint(1, max_dist)
    x_μ_dict = [random.uniform(0.2,0.8) for i in range(n_dist)]
    y_μ_dict = [random.uniform(0.2,0.8) for i in range(n_dist)]
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
        V = fd.FunctionSpace(mesh, "CG", 2)
        F = fd.Function(V)
        F.interpolate(source)
        return F
    return func