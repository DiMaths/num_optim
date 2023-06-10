from abc import abstractmethod, ABC
from typing import Union

import numpy as np

import func
from newton_family import NewtonFamily


class QuasiNewtonMethod(NewtonFamily, ABC):
    def __init__(self, f: func.Function, name: str, start_point: np.ndarray, norm: Union[str, float], eps: float,
                 max_iterations: int, initial_alpha: float, rho: float, c: float):
        super().__init__(f, name, start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        self.H = np.eye(self.f.get_dim()) # inital approx. of inverse Hessian

    def compute_b(self):
        x_k = self.x_k
        grad_f_k = self.f.grad(x_k)
        self.p_k = -self.H @ grad_f_k  # search direction (6.18)


        alpha_k = self.compute_alpha_k()  # backtracking line search


        x_new = x_k + alpha_k * self.p_k  # (6.3)
        # define s_k and y_k (6.5)
        s_k = x_new - x_k  # or: alpha_k * p_k #for sk1?
        grad_f_new = self.f.grad(x_new)
        y_k = grad_f_new - grad_f_k

        self.H = self.approx_inverse_hessian(y_k, s_k)

        self.x_k = x_new
        self.grad_f_k = grad_f_new
        return self.H

    @abstractmethod
    def approx_inverse_hessian(self, y_k: np.array, s_k: np.array) -> np.array:
        raise NotImplementedError



class SR1(QuasiNewtonMethod):
    def __init__(self, f: func.Function, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 1e-6, max_iterations: int = 1e6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.6):
        super().__init__(f, "SR1", start_point, norm, eps, max_iterations, initial_alpha, rho, c)

    def approx_inverse_hessian(self, y_k, s_k):
        if np.abs(np.dot(y_k, s_k)) >= self.eps * np.linalg.norm(y_k) * np.linalg.norm(s_k):
            Hy = np.dot(self.H, y_k)
            if np.dot(y_k, s_k - Hy) != 0:
                return self.H + np.outer(s_k - Hy, s_k - Hy) / np.dot(y_k, s_k - Hy)
        return self.H


class BFGS(QuasiNewtonMethod):
    def __init__(self, f: func.Function, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                 eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                 c: float = 0.6):
        super().__init__(f, "BFGS", start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        
    def approx_inverse_hessian(self, y_k: np.array, s_k: np.array) -> np.array:
        import numpy.linalg as la
        I = np.eye(self.f.get_dim())

        if y_k.T @ s_k > 0:# http://www2.imm.dtu.dk/documents/ftp/publlec/lec2_99.pdf
            # (6.17) compute H_{k+1} -> H_new using BFGS formula
            rho_k = 1.0 / (y_k.T @ s_k)  # (6.14)
            return (I - rho_k * s_k @ y_k.T) @ self.H @ (I - rho_k * y_k @ s_k.T) + rho_k * s_k @ s_k.T  # add inv
        else:
            return self.H


class SR1_TR(NewtonFamily):
    def __init__(self, f: func.Function, start_point: np.ndarray = None, norm: Union[str, float] = 2,
                                   eps: float = 10 ** -6, max_iterations: int = 10 ** 6, initial_alpha: float = 1, rho: float = 0.99,
                                   c: float = 0.6):
        super().__init__(f, "SR1_TR", start_point, norm, eps, max_iterations, initial_alpha, rho, c)
        self.delta_k: float = 0.1 # for trust region SR1 use 0.1 as initial trust-region radius
            
    def compute_s(self, grad_f_k, B_k, delta_k):
        # p_k = -np.linalg.inv(B_k) @ grad_f_k
        p_k = np.linalg.solve(B_k, -grad_f_k) #does the same without compiting inv
        norm_p_k = np.linalg.norm(p_k)
        if norm_p_k <= delta_k:
            s = p_k
        else:
            s = (delta_k * p_k )/ norm_p_k  # if norm of p_k outside trust region radius, scale down p_k
        return s

    def compute_b(self):
        x_k = self.x_k
        B_k = self.f.hessian(x_k)
        f_k = self.f.evaluate(x_k)
        grad_f_k = self.f.grad(x_k)
        eta = 0.15# 1e-3
        delta_k = self.delta_k

        # compute s_k (6.27)
        # solve for s https://digital.library.unt.edu/ark:/67531/metadc283525/m2/1/high_res_d/metadc283525.pdf
        s_k = self.compute_s(grad_f_k, B_k, delta_k)

        y_k = self.f.grad(x_k + s_k) - grad_f_k
        ared = f_k - self.f.evaluate(x_k + s_k)
        pred = -(grad_f_k.T @ s_k + 1 / 2 * s_k.T @ B_k @ s_k)

        if pred != 0.0:
            if ared / pred > eta:
                x_new = x_k + s_k
            else:
                x_new = x_k
            if ared / pred > 0.75:
                if np.linalg.norm(s_k) <= 0.8 * delta_k:
                    delta_new = delta_k
                else:
                    delta_new = 2 * delta_k
            elif 0.1 <= ared / pred <= 0.75:
                delta_new = delta_k
            else:
                delta_new = 0.5 * delta_k
        else:
            x_new = x_k
            delta_new = delta_k

        # if (6.26) holds:
        yBs = y_k - B_k @ s_k
        if yBs.T @ s_k != 0.0:
            if np.abs(s_k.T @ yBs) >= delta_k * np.linalg.norm(s_k) * np.linalg.norm(yBs):
                # use (6.24) to compute B_{k+1}
                B_k = B_k + (yBs @ yBs.T) / (yBs.T @ s_k)

        self.x_k = x_new
        self.grad_f_k = self.f.grad(x_new)
        self.delta_k = delta_new

        return B_k
