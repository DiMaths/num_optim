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
                 eps: float = 10**-6,
                 max_iterations: int = 10 ** 6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.99):
        self.name = name
        self.f = f
        if start_point is not None:
            self.x_k = start_point
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
        print('-' * 50)

    @abstractmethod
    def update(self):
        self.iterations += 1
        # optional printing for seeing the process
        if self.iterations % 10**(np.floor(np.log10(self.iterations))) == 0:
            # if (self.iterations < 100_000 and self.iterations % 10_000 == 0) or self.iterations % 100_000 == 0:
            print(f"{self.iterations} iterations, residual's norm = {self.residual_norm()}")

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

class ConjugateGradient(LineSearch):
    def __init__(self, f: func.Function,
                 name: str = 'Linear',
                 start_point: np.ndarray = None,
                 norm: Union[str, float] = 2,
                 eps: float = 10**-6,
                 max_iterations: int = 10 ** 6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.99):
        # 'Linear', 'P-R', 'F-R'

        if name not in ['Linear', 'P-R', 'F-R']:
            raise ValueError("Possible names for NewtonFamilyMethod are: 'Linear', 'P-R', 'F-R'")
        super().__init__(f, 'CG_' + name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        self.r_k = np.dot(self.f.A, self.x_k) - self.f.b
        self.p_k = -self.r_k
        self.r_k_norm = None

    def compute_alpha_k(self) -> float:
        raise NotImplementedError("We don't use compute alpha for cg so far")

    def update(self):
        self.r_k_norm = np.dot(self.r_k.T, self.r_k)
        self.alpha_k = self.r_k_norm / np.dot(self.p_k.T, self.f.A.dot(self.p_k))

        self.r_k = self.r_k + self.alpha_k * np.dot(self.f.A, self.p_k)

        # now self.r_k is r_k_next, r_k norm however is r_k.T @ r_k
        beta_k_next = np.dot(self.r_k.T, self.r_k) / self.r_k_norm

        self.x_k = self.x_k + self.alpha_k * self.p_k
        self.p_k = -self.r_k + beta_k_next * self.p_k
        super().update()

    def residual_norm(self) -> float:
        return np.linalg.norm(self.r_k, ord=2)



