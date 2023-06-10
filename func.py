from abc import abstractmethod, ABC

import numpy as np


class Function(ABC):
    def __init__(self, dim: int, num_mode: bool = True, eps: float = 1e-6):
        """
        Base class for generalized function definitions
        :param dim: dimensions of function
        :param num_mode: True to use numerical approximations of gradients and hessian
        :param eps: deviation from point used to approximate grad, for hessian there's an extra multiplier
        """
        self.dim = dim
        self.num_mode = num_mode
        self.eps = eps

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        evaluation at point, differs for each subclass
        :param x: n-dimensional point x
        :return: 1-dimensional function value
        """
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        if self.num_mode:
            return self.num_grad(x, self.eps)
        else:
            return self.explicit_grad(x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        if self.num_mode:
            # added 0.01 multiplier to increase accuracy (trying to counter the error from numerical grad)
            return self.num_hessian(x, 0.01 * self.eps)
        else:
            return self.explicit_hessian(x)

    @abstractmethod
    def explicit_grad(self, x: np.ndarray) -> np.ndarray:
        """
        explicit gradient, differs for each subclass
        :param x: n-dimensional point x
        :return: n-dimensional gradient
        """
        raise NotImplementedError

    @abstractmethod
    def explicit_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        explicit hessian, differs for each subclass
        :param x: n-dimensional point x
        :return: nxn dimensional hessian-matrix
        """
        raise NotImplementedError

    def num_grad(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        calculate numerical gradient aprox. at point
        :param x: n-dimensional point x
        :param eps: deviation param for approximation
        :return: n-dimensional gradient
        """
        grad = []
        if self.get_dim() == 1:
            grad = 0.5 * (self.evaluate(x+eps) - self.evaluate(x-eps)) / eps
        else:
            eps_vecs = eps * np.diag(np.ones(shape=self.dim))
            for eps_vec in eps_vecs:
                temp_partial = 0.5 * (self.evaluate(x+eps_vec) - self.evaluate(x-eps_vec)) / eps
                # temp_partial = (self.evaluate(x + eps_vec) - self.evaluate(x)) / eps
                grad.append(temp_partial)
        return np.array(grad)

    def num_hessian(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        calculate numerical hessian approx. at point
        :param x: n-dimensional point x
        :param eps: deviation param for approximation
        :return: nxn dimensional hessian-matrix
        """
        hess = []
        if self.get_dim() == 1:
            hess = 0.5 * (self.num_grad(x+eps) - self.num_grad(x-eps)) / eps
        else:
            eps_vecs = eps * np.diag(np.ones(shape=self.dim))
            for eps_vec in eps_vecs:
                temp_partial = 0.5 * (self.num_grad(x+eps_vec) - self.num_grad(x-eps_vec)) / eps
                # temp_partial = (self.num_grad(x + eps_vec) - self.num_grad(x)) / eps
                hess.append(temp_partial)
        return np.array(hess)

    def get_dim(self) -> int:
        return self.dim


class Quadratic(Function):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        :param A: A is positive definite square matrix
        :param b:  b is column vector
        """
        super().__init__(dim=len(b.T))
        self.A = A
        self.b = b

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x.T @ self.A @ x - self.b.T @ x

    def explicit_grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def explicit_hessian(self, x: np.ndarray) -> np.ndarray:
        return self.A


class Sin(Function):
    def __init__(self):
        super().__init__(dim=1)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def explicit_grad(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def explicit_hessian(self, x: np.ndarray) -> np.ndarray:
        return -np.sin(x)


class UnivariatePolynomial(Function):
    def __init__(self, degree: int, coeffs: np.ndarray = None):
        """
        Generate univar. Poly
        :param degree: degree of poly
        :param coeffs: coefficients of polynomial, if None Random [-5,5]
        """
        super().__init__(dim=1)
        self.degree = degree
        if coeffs is not None:
            if len(coeffs) == self.degree + 1:
                self.coeffs = coeffs
            else:
                raise ValueError(f" received { {len(coeffs)} } coefficients, "
                                 f"but degree of polynomial is {self.degree}, need {self.degree+1} coefficients")
        else:
            self.coeffs = np.random.randint(-5, 5, self.degree+1, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.coeffs @ np.vander(x, self.degree+1).T

    def explicit_grad(self, x: np.ndarray) -> np.ndarray:
        grad_coeffs = np.array([(self.degree - i)*self.coeffs[i] for i in range(len(self.coeffs)-1)])
        return grad_coeffs @ np.vander(x, self.degree).T

    def explicit_hessian(self, x: np.ndarray) -> np.ndarray:
        hessian_coeffs = np.array([(self.degree - i) * (self.degree - i - 1) * self.coeffs[i] for i in range(len(self.coeffs) - 2)])

        return hessian_coeffs @ np.vander(x, self.degree-1).T


class MultivariateLinear(Function):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Multivariate linear function
        :param A: n-dimensional k-values
        :param b:  n-dimensional d-values
        """
        super().__init__(dim=len(b.T))
        self.A = A
        self.b = b

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def explicit_grad(self, x: np.ndarray) -> np.ndarray:
        return self.A.T

    def explicit_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(shape=(len(self.A), len(self.A)))


class RosenBrock(Function):
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return 100 * ((x[1] - x[0] ** 2) ** 2) + (1 - x[0]) ** 2

    def explicit_grad(self, x: np.ndarray) -> np.ndarray:
        return np.array([400 * x[0] * (x[0] ** 2 - x[1]) - 2 * (-x[0] + 1),
                         200 * (-x[0] ** 2 + x[1])])

    def explicit_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[800 * x[0] ** 2 - 400 * (-x[0] ** 2 + x[1]) + 2, -400 * x[0]],
                         [-400 * x[0], 200]])


class SecondObjective(Function):
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2

    def explicit_grad(self, x: np.ndarray) -> np.ndarray:
        return np.array([300 * x[0] * x[1] ** 2 + 0.5 * x[0] + 2 * x[1] - 2,
                         300 * x[0] ** 2 * x[1] + 8 * (0.25 * x[0] + x[1] - 1)])

    def explicit_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[300 * x[1] ** 2 + 0.5, 600 * x[0] * x[1] + 2],
                         [600 * x[0] * x[1] + 2, 300 * x[0] ** 2 + 8]])
