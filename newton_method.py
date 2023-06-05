from abc import abstractmethod, ABC
from typing import Union

import numpy as np
from numpy.linalg import LinAlgError

import func
from optim_algos import LineSearch


class NewtonFamily(LineSearch, ABC):
    def __init__(self, f: func.Function, name: str, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)

    def compute_p_k(self):
        return - self.compute_b() @ self.grad_f_k

    def update(self):
        if self.iterations == 0 or self.alpha_k == 1 or (self.grad_f_k != self.f.grad(self.x_k)).any():
            # when next grad is the same as current, but alpha isn't 1, we are stuck
            # it could happen that the optimal step size is more than -1*grad, but than alpha should be 1
            self.grad_f_k = self.f.grad(self.x_k)
            self.p_k = self.compute_p_k()
            self.alpha_k = self.compute_alpha_k()

            self.x_k += self.alpha_k * self.p_k
            super().update()
        else:
            self.stuck = True

    @abstractmethod
    def compute_b(self) -> np.array:
        raise NotImplementedError


class NewtonMethod(NewtonFamily):
    def compute_b(self):
        if np.allclose(self.f.num_hessian(self.x_k), np.zeros(shape=self.f.get_dim())):
            if np.allclose(self.f.hessian(self.x_k), np.zeros(shape=self.f.get_dim())):
                print("DROPPED TO SD")
                B_k = np.diag(np.ones(shape=self.f.get_dim()))
            else:
                print("NUM ISSUES WITH HESSIAN")
                if self.f.get_dim() > 1:
                    B_k = np.linalg.inv(self.f.hessian(self.x_k))
                else:
                    B_k = np.array([1 / self.f.hessian(self.x_k)])
        else:
            try:
                if self.f.get_dim() > 1:
                    B_k = np.linalg.inv(self.f.num_hessian(self.x_k))
                else:
                    B_k = np.array([1 / self.f.num_hessian(self.x_k)])
            except LinAlgError:
                print("DROPPED TO SD")
                B_k = np.diag(np.ones(shape=self.f.get_dim()))
        return B_k

    def __init__(self, f: func.Function, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, "Newton Method", start_point, norm, eps, max_iterations, initial_alpha, rho, c)


class SteepestDescent(NewtonFamily):
    def compute_b(self):
        return np.diag(np.ones(shape=self.f.get_dim()))

    def __init__(self, f: func.Function, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, "Steepest Descent", start_point, norm, eps, max_iterations, initial_alpha, rho, c)
