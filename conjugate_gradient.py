import numpy as np
from typing import Union


from optim_algos import LineSearch
import func


class LinearConjugateGradient(LineSearch):
    def __init__(self, f: func.Function,
                 start_point: np.ndarray = None,
                 norm: Union[str, float] = 2,
                 eps: float = 10**-6,
                 max_iterations: int = 10 ** 6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.99):

        super().__init__(f, "CG_Linear", start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        self.r_k = self.f.A @ self.x_k - self.f.b
        self.p_k = -self.r_k
        self.r_k_norm = None

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


class NonlinearConjugateGradient(LineSearch):
    def __init__(self, f: func.Function,
                 name: str = 'FR',
                 start_point: np.ndarray = None,
                 norm: Union[str, float] = 2,
                 eps: float = 10 ** -6,
                 max_iterations: int = 10 ** 6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.5):
        super().__init__(f, "CG_" + name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        self.grad_f_k_next = None
        self.p_k = -self.grad_f_k

    def update(self):
        self.alpha_k = self.compute_alpha_k()
        self.x_k += self.alpha_k * self.p_k
        self.grad_f_k_next = self.f.grad(self.x_k)
        beta_k_next = self.compute_beta_next()
        self.p_k = -self.grad_f_k_next + beta_k_next * self.p_k
        self.grad_f_k = self.grad_f_k_next
        super().update()

    def compute_beta_next(self):
        if self.name == 'CG_PR':
            return (self.grad_f_k_next.T @ (self.grad_f_k_next - self.grad_f_k)) / (self.grad_f_k.T @ self.grad_f_k)
        elif self.name == 'CG_FR':
            return (self.grad_f_k_next.T @ self.grad_f_k_next) / (self.grad_f_k.T @ self.grad_f_k)