from abc import abstractmethod, ABC
from typing import Union

import numpy as np

import func
from newton_method import NewtonFamily


class QuasiNewtonMethod(NewtonFamily, ABC):
    def compute_b(self):
        x_k = self.x_k
        grad_f_k = self.f.grad(x_k)
        self.p_k = -self.H @ grad_f_k  # search direction (6.18)

        alpha_k = self.compute_alpha_k()  # backtracking line search # same for SR1 ?

        x_new = x_k + alpha_k * self.p_k  # (6.3)
        # define s_k and y_k (6.5)
        s_k = x_new - x_k  # or: alpha_k * p_k #for sk1?
        grad_f_new = self.f.grad(x_new)
        y_k = grad_f_new - grad_f_k

        self.H = self.approx_inverse_hessian(y_k, s_k)

        self.x_k = x_new
        self.grad_f_k = grad_f_new
        return self.H

    def __init__(self, f: func.Function, name: str, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        self.H = np.eye(self.f.get_dim())  # inital approx. of inverse Hessian - added ???

    @abstractmethod
    def approx_inverse_hessian(self, y_k: np.array, s_k: np.array) -> np.array:
        raise NotImplementedError


class Sr1(QuasiNewtonMethod):
    def approx_inverse_hessian(self, y_k, s_k):
        if np.abs(np.dot(y_k, s_k)) >= self.eps * np.linalg.norm(y_k) * np.linalg.norm(s_k):
            Hy = np.dot(self.H, y_k)
            return self.H + np.outer(s_k - Hy, s_k - Hy) / np.dot(y_k, s_k - Hy)
        else:
            return self.H

    def __init__(self, f: func.Function, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, "Sr1", start_point, norm, eps, max_iterations, initial_alpha, rho, c)


class BFGS(QuasiNewtonMethod):
    def approx_inverse_hessian(self, y_k: np.array, s_k: np.array) -> np.array:
        if np.abs(y_k @ s_k) >= 0:  # (6.7) check curvature condition?
            # (6.17) compute H_{k+1} -> H_new using BFGS formula
            I = np.eye(self.f.get_dim())
            rho_k = 1.0 / (y_k.T @ s_k)  # (6.14)
            return (I - rho_k * s_k @ y_k.T) @ self.H @ (I - rho_k * y_k @ s_k.T) + rho_k * s_k @ s_k.T
        else:
            return self.H

    def __init__(self, f: func.Function, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, "BFGS", start_point, norm, eps, max_iterations, initial_alpha, rho, c)

