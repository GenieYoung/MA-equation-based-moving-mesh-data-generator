import firedrake as fd
from animate import RiemannianMetric, recover_gradient_l2
from ufl.core.expr import Expr

def monitor_func1(u_func):
    def monitor(mesh):
        u_func_mesh = u_func(mesh)

        if isinstance(u_func_mesh, Expr) and  \
                not isinstance(u_func_mesh, fd.function.Function) and \
                not isinstance(u_func_mesh, fd.Function):
            H = fd.ufl.grad(fd.ufl.grad(u_func_mesh))
        elif u_func_mesh.function_space().ufl_element().degree() <= 1:
            TV = fd.TensorFunctionSpace(mesh, "CG", 1)
            H = RiemannianMetric(TV)
            H.compute_hessian(u_func_mesh, method="L2")
        else:
            #H = fd.grad(fd.grad(u_func_mesh))
            TV = fd.TensorFunctionSpace(mesh, "CG", 1)
            H = RiemannianMetric(TV)
            H.compute_hessian(u_func_mesh, method="L2")

        V = fd.FunctionSpace(mesh, "CG", 1)
        Hnorm = fd.Function(V, name="Hnorm")
        Hnorm.interpolate(fd.sqrt(fd.inner(H, H)))
        Hnorm_max = Hnorm.dat.data.max()
        Hnorm_max = 1 if Hnorm_max == 0 else Hnorm_max
        m = 1 + 10 * Hnorm / Hnorm_max
        return m
    return monitor

def monitor_func2(u_func):
    def monitor(mesh):
        V = fd.FunctionSpace(mesh, "CG", 1)
        Gnorm = fd.Function(V, name="Gnorm")
        Gnorm.interpolate(fd.ufl.sqrt(1+0.1*fd.ufl.inner(fd.grad(u_func(mesh)), fd.grad(u_func(mesh)))))
        Gnorm_max = Gnorm.dat.data.max()
        m = 1 + 10 * Gnorm / Gnorm_max
        return m
    return monitor

def monitor_func3(u_func):
    def monitor(mesh):
        V = fd.FunctionSpace(mesh, "CG", 2)
        Gnorm = fd.Function(V, name="Gnorm")
        grad = recover_gradient_l2(u_func(mesh))
        Gnorm.interpolate(fd.ufl.sqrt(1+0.1*fd.ufl.inner(grad, grad)))
        Gnorm_max = Gnorm.dat.data.max()
        return 1 + 10 * Gnorm / Gnorm_max
    return monitor

def monitor_func4(u_func):
    def monitor(mesh):
        V = fd.FunctionSpace(mesh, "CG", 1)
        TV = fd.TensorFunctionSpace(mesh, "CG", 1)
        H = RiemannianMetric(TV)
        H.compute_hessian(u_func(mesh), method="L2")
        Hnorm = fd.Function(V, name="Hnorm")
        Hnorm.interpolate(fd.sqrt(fd.inner(H, H)))
        Hnorm_max = Hnorm.dat.data.max()
        m = 1 + 10 * Hnorm / Hnorm_max
        return m
    return monitor