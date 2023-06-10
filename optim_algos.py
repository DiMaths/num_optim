from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.linalg import LinAlgError

import func


class LineSearch(ABC):
    def __init__(self, f: func.Function,
                 name: str,
                 start_point: np.ndarray = None,
                 norm: Union[str, float] = 2,
                 eps: float = 1e-6,
                 max_iterations: int = 1e6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.99):
        """
        :param f: func.Function, function to perform line search on
        :param name: str, name of the algo
        :param start_point: np.ndarray, starting point = the 0th iterate = x_0
        :param norm: Union[str, float] to use as ord param in np.norm()
        :param eps: float, stopping threshold for residual's norm
        :param max_iterations: int, upper bound on k
        :param initial_alpha: float, initial alpha to check Wolfe Conditions
        :param rho: float in (0, 1) - multiplier for updating alpha
        :param c: float in (0, 1) - constant for Wolfe Conditions
        """
        self.name = name
        self.f = f
        if start_point is not None:
            self.x_k = start_point.copy()
        else:
            self.x_k = np.zeros(shape=f.get_dim())

        self.initial_alpha = initial_alpha
        self.rho = rho
        self.c = c
        self.norm = norm
        self.eps = eps
        self.max_iterations = max_iterations

        self.p_k = None
        self.alpha_k = None
        self.iterations = 0
        self.stuck = False

        self.grad_f_k = self.f.grad(self.x_k)

    def compute_alpha_k(self) -> float:
        alpha_k = self.initial_alpha
        f_k = self.f.evaluate(self.x_k)
        while self.f.evaluate(self.x_k + alpha_k*self.p_k) > f_k + self.c * alpha_k * self.grad_f_k.T @ self.p_k:
            alpha_k *= self.rho
        return alpha_k

    def print_solution(self, cut_off: bool = False):
        print('-' * 50)
        if self.stuck:
            print('PROBLEMS WITH FORMAT PRECISION')
        if cut_off:
            print("!!!THE EXECUTION IS CUT OFF!!!")
        print(f"Execution took {self.iterations} iterations")
        print(f"Found solution is x = {self.x_k}")
        print(f"Final residual norm is {self.residual_norm()}")

    @abstractmethod
    def update(self):
        self.iterations += 1
        """if self.iterations % 10**(np.floor(np.log10(self.iterations))) == 0:
            print(f"{self.iterations} iterations, residual's norm = {self.residual_norm()}")"""

    def execute(self):
        print(f"Execute {self.name}")
        self.iterations = 0

        while self.stuck is False and\
                self.iterations < self.max_iterations and self.residual_norm() > self.eps:
            self.update()

        if self.iterations == self.max_iterations:
            self.print_solution(cut_off=True)
        else:
            self.print_solution()

    def residual_norm(self):
        return np.linalg.norm(self.grad_f_k, ord=self.norm)
