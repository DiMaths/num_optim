from typing import Union

import func
from optim_algos import LineSearch
import numpy as np


class Sk1(LineSearch):
    def __init__(self, f: func.Function, name: str, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.99):
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)

        # initalize bk as identity matrix
        self.Bk = np.eye(self.f.get_dim(), dtype=float)

    def linesearch(self, dk: np.array):
        # impl. Armijo-Goldstein line search

        alpha = self.initial_alpha
        c = self.c
        rho = self.rho

        # Get current x value and gradient
        xk = self.x_k
        gk = self.f.grad(xk)
        fk = self.f.evaluate(xk)

        # Perform line search
        while alpha > self.eps:
            x_new = xk + alpha * dk
            fk_new = self.f.evaluate(x_new)
            if fk_new <= fk + c * alpha * np.dot(gk, dk):
                return alpha
            alpha *= rho

        return alpha

    def update(self):
        # get current x value
        xk = self.x_k
        # gradient at pont x
        gk = self.f.grad(xk)
        # todo maybe replace this solve with sherman-morrison formula
        dk = -1.0 * np.linalg.solve(self.Bk, gk)

        # perform linesearch to find step_size
        sk = self.linesearch(dk)

        # update x value
        self.x_k = xk + sk * dk
        sk = self.x_k - xk
        yk = self.f.grad(self.x_k) - self.f.grad(xk)

        ys = yk - np.dot(self.Bk, sk)
        # todo this line might be more efficient
        # todo is greater equal really right here?
        if np.dot(ys.T, sk) > 0:
            self.Bk = self.Bk - (ys * ys.T) / np.dot(ys.T, sk)

        # set stuff for superclass monitoring and update
        self.grad_f_k = gk
        super().update()
