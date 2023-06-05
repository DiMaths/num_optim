from optim_algos import LineSearch
import func
import numpy as np
from typing import Union

# later within NewtonFamilyMethod bc circular import not possible ???
class Bfgs(LineSearch):
    def __init__(self, f: func.Function, name: str, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10**-6, max_iterations: int = 10**6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        self.H = np.eye(self.f.get_dim())  # inital approx. of inverse Hessian

    def bfgs_approx(self):
        x_k = self.x_k
        grad_f_k = self.f.grad(x_k)
        self.p_k = -(self.H @ grad_f_k)  # search direction

        alpha_k = LineSearch.compute_alpha_k(self)  # backtracking line

        x_new = x_k + alpha_k * self.p_k
        s_k = x_new - x_k  # or: alpha_k * p_k
        grad_f_new = self.f.grad(x_new)
        y_k = grad_f_new - grad_f_k

        # (6.17) compute H_{k+1} -> H_new using BFGS formula
        rho_k = 1.0/(y_k.T @ s_k)
        I = np.eye(self.f.get_dim())
        H_new = (I - rho_k * s_k @ y_k.T) @ self.H @ (I - rho_k * y_k @ s_k.T) + rho_k * s_k @ s_k.T

        self.x_k = x_new
        self.grad_f_k = grad_f_new
        self.H = H_new
        super().update()

