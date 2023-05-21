import numpy as np


class Function:
    def __init__(self, dim: int = None):
        self.dim = dim

    def evaluate(self, x: np.ndarray):
        # evaluation at point, differs for each subclass
        raise NotImplementedError

    def grad(self, x: np.ndarray):
        # explicit gradient, differs for each subclass
        raise NotImplementedError

    def hessian(self, x: np.ndarray):
        # explicit hessian, differs for each subclass
        raise NotImplementedError

    def num_grad(self, x: np.ndarray, eps: float = 10**-6):
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

    def num_hessian(self, x: np.ndarray, eps: float = 10**-8):
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

    def get_dim(self):
        return self.dim


class Quadratic(Function):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        # A is positive definite square matrix, b is column vector
        super().__init__(dim=len(b.T))
        self.A = A
        self.b = b

    def evaluate(self, x: np.ndarray):
        return 0.5 * x.T @ self.A @ x - self.b.T @ x

    def grad(self, x: np.ndarray):
        return self.A @ x - self.b

    def hessian(self, x: np.ndarray):
        return self.A


class Sin(Function):
    def __init__(self):
        super(Sin, self).__init__(dim=1)

    def evaluate(self, x: float):
        return np.sin(x)

    def grad(self, x: float):
        return np.cos(x)

    def hessian(self, x: float):
        return -np.sin(x)


class UnivariatePolynomial(Function):
    def __init__(self, degree: int, coeffs: np.ndarray = None):
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

    def evaluate(self, x: float):
        return np.array([self.coeffs.T @ create_vandermonde_vector(x, self.degree)])

    def grad(self, x: float):
        grad_coeffs = np.array([(self.degree - i)*self.coeffs[i] for i in range(len(self.coeffs)-1)])
        return np.array([grad_coeffs.T @ create_vandermonde_vector(x, self.degree-1)])

    def hessian(self, x: float):
        hessian_coeffs = np.array([(self.degree - i) * (self.degree - i - 1) * self.coeffs[i] for i in range(len(self.coeffs) - 2)])

        return np.array([hessian_coeffs.T @ create_vandermonde_vector(x, self.degree-2)])


def create_vandermonde_vector(x: float, degree: int) -> np.ndarray:
    # creates x^degree, x^(degree-1), .... x^2, x, 1
    return np.array([float(x) ** i for i in range(degree+1)][::-1])


class MultivariateLinear(Function):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        super().__init__(dim=len(b.T))
        self.A = A
        self.b = b

    def evaluate(self, x: np.ndarray):
        return self.A @ x - self.b

    def grad(self, x: np.ndarray):
        return self.A.T

    def hessian(self, x: np.ndarray):
        return np.zeros(shape=(len(self.A), len(self.A)))
