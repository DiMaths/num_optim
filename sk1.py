from typing import Union

import func
from optim_algos import LineSearch
import numpy as np


class Sk1(LineSearch):
    def __init__(self, f: func.Function, name: str, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 0.3, rho: float = 0.8,
                 c: float = 0.99):
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)

        # initalize bk as identity matrix
        self.H = np.eye(self.f.get_dim(), dtype=float)

    def linesearch(self, dk: np.array, xk: np.array, gk: np.array):
        alpha = self.initial_alpha
        while self.f.evaluate(xk + alpha * dk) > self.f.evaluate(xk) + self.initial_alpha * alpha * np.dot(gk, dk):
            alpha *= self.rho
        return alpha

    def update(self):
        # todo in this implementation of sk1 I use a Hessian approximation
        #   this might be implemented with approximating a Broyden matrix as well!?!?

        xk = self.x_k
        # gradient at pont x
        gk = self.f.grad(xk)
        dk = -np.dot(self.H, gk)

        # perform linesearch
        alpha = self.linesearch(dk, xk, gk)

        # update x value
        x_new = xk + alpha * dk
        yk = self.f.grad(x_new) - gk

        s = alpha * dk
        if np.abs(np.dot(yk, s)) >= self.eps * np.linalg.norm(yk) * np.linalg.norm(s):
            Hy = np.dot(self.H, yk)
            self.H += np.outer(s - Hy, s - Hy) / np.dot(yk, s - Hy)

        self.x_k = x_new
        # set stuff for superclass monitoring and update
        self.grad_f_k = gk
        super().update()
