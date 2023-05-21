import numpy as np
import scipy.linalg
import func
import optim_algos as alg

if __name__ == '__main__':

    A = np.array([[1, 0, 1],
                  [0, 0.5, 1],
                  [1, 1, 1]])
    # A = scipy.linalg.hilbert(n=20)
    b = np.array([3, 1, 2]).T
    # b = np.ones(20).T
    ax_b = func.MultivariateLinear(A=A, b=b)
    cg_linear = alg.ConjugateGradient(ax_b)
    cg_linear.execute()

    for n in [2]:
        print('*' * 25)
        print(f"HILBERT MATRIX {n} x {n}")
        print('*' * 25)
        A = scipy.linalg.hilbert(n=n)
        hilbert_quadratic = func.Quadratic(A=A, b=np.ones(n))
        sd_hilbert = alg.NewtonFamilyMethod(hilbert_quadratic,
                                            name='SD',
                                            max_iterations=10**6)

        x_star = np.linalg.solve(A, np.ones(n))
        sd_hilbert.execute()

        norm_of_difference = scipy.linalg.norm(sd_hilbert.x_k - x_star, ord=2)
        print(f"Exact solution = {x_star}")
        print(f"Norm of the difference between exact and found solutions = {norm_of_difference},"
              f" between starting point and exact solution = {scipy.linalg.norm(x_star, ord=2)}")
        print('**'*15)

    print("/" * 50)
    print('Task 1 and 6, polynomial of degree 4')
    print("f(x) = 0.25x^4  +(2/3)x^3 -0.5x^2 -2x -7")
    print("f'(x) = (x+1)(x-1)(x+2) --> minimizers are x= -2 and 1")
    poly1 = func.UnivariatePolynomial(degree=4, coeffs=np.array([0.25, float(2 / 3), -0.5, -2., -7]))
    sd1 = alg.NewtonFamilyMethod(poly1, start_point=0, name='SD')
    sd1.execute()
    nm1 = alg.NewtonFamilyMethod(poly1, start_point=0, name='NM')
    nm1.execute()

