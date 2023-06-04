from typing import Union

import numpy as np
from numpy.linalg import LinAlgError

import func


class LineSearch:
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


class NewtonFamilyMethod(LineSearch):
    def __init__(self, f: func.Function,
                 name: str = 'SD',
                 start_point: np.ndarray = None,
                 norm: Union[str, float] = 2,
                 eps: float = 10**-6,
                 max_iterations: int = 10 ** 6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.99):
        # SD - Steepest Descent
        # NM - Newton Method
        # QN - Quasi-Newton Method
        if name not in ['SD', 'QN', 'NM']:
            raise ValueError("Possible names for NewtonFamilyMethod are: 'SD', 'NM' and 'QN'")
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)

    def compute_B(self):
        if self.name == 'SD':
                B_k = np.diag(np.ones(shape=self.f.get_dim()))

        elif self.name == 'NM':
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

            """if self.f.get_dim() > 1:
                try:
                    B_k = np.linalg.inv(self.f.num_hessian(self.x_k))
                except LinAlgError:
                    print(f"Problems with inverting numerical hessian at point x_k = {self.x_k},"
                          f" which is {self.f.num_hessian(self.x_k)}")
                    try:
                        B_k = np.linalg.inv(self.f.hessian(self.x_k))
                    except LinAlgError:
                        print(f"Explicit hessian failed at point x_k = {self.x_k},"
                              f" which is {self.f.hessian(self.x_k)}")
                        raise LinAlgError"""
            """else:
                try:
                    B_k = np.array([1 / self.f.num_hessian(self.x_k)])
                except ZeroDivisionError:
                    print(f"Problems with inverting numerical hessian at point x_k = {self.x_k},"
                          f" which is {self.f.num_hessian(self.x_k)}")
                    try:
                        B_k = np.array([1 / self.f.hessian(self.x_k)])
                    except ZeroDivisionError:
                        print(f"Explicit hessian failed at point x_k = {self.x_k},"
                              f" which is {self.f.hessian(self.x_k)}")
                        raise LinAlgError"""
        else:
            # QN
            raise NotImplementedError("Quasi Newton Method is not implemented yet")

        return B_k

    def compute_p_k(self):
        return - self.compute_B() @ self.grad_f_k

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


class LinearConjugateGradient(LineSearch):
    def __init__(self, f: func.Function,
                 name: str = 'CG_Linear',
                 start_point: np.ndarray = None,
                 norm: Union[str, float] = 2,
                 eps: float = 10**-6,
                 max_iterations: int = 10 ** 6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.99):

        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)
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
                 name: str = 'CG_FR',
                 start_point: np.ndarray = None,
                 norm: Union[str, float] = 2,
                 eps: float = 10 ** -6,
                 max_iterations: int = 10 ** 6,
                 initial_alpha: float = 1,
                 rho: float = 0.99,
                 c: float = 0.5):
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)
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


