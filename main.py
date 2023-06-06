import numpy as np
import scipy.linalg

import func
import conjugate_gradient as cg
import newton_family

if __name__ == '__main__':
    # Nonlinear CG both F-R and P-R

    fs = [func.RosenBrock(2), func.SecondObjective(2)]
    fs_names = ["Rosenblock", "Alternative"]
    points = [[np.array([1.2, 1.2]), np.array([-1.2, 1.]), np.array([0.2, 0.8])],
              [np.array([-0.2, 1.2]), np.array([3.8, 0.1]), np.array([1.9, 0.6])]]
    names = ["FR", "PR"]

    for i, f in enumerate(fs):
        print()
        print('|' * 50)
        print(f'Experiments on "{fs_names[i]}" function')
        print('|' * 50)
        for point in points[i]:
            print(f'***** STARTING POINT = {point} *****')
            for name in names:
                test_alg = cg.NonlinearConjugateGradient(f, name, point)
                test_alg.execute()
                print()

    print("*"*50)
    # Linear CG task
    for n in [5, 8, 12, 20, 30]:
        print('*' * 25)
        print(f"HILBERT MATRIX {n} x {n}")
        print('*' * 25)
        A = scipy.linalg.hilbert(n=n)
        x_star = np.linalg.solve(A, np.ones(n))

        hilbert_quadratic = func.Quadratic(A=A, b=np.ones(n))
        sd_hilbert = newton_family.SteepestDescent(hilbert_quadratic, max_iterations=10 ** 5)
        sd_hilbert.execute()

        norm_of_difference = scipy.linalg.norm(sd_hilbert.x_k - x_star, ord=2)
        print(f"Exact solution = {x_star}")
        print(f"Norm of the difference between exact and found solutions = {norm_of_difference},")
        print(f" between starting point and exact solution = {scipy.linalg.norm(x_star, ord=2)}")
        print('**' * 15)

        cg_linear = cg.LinearConjugateGradient(hilbert_quadratic)
        cg_linear.execute()

        norm_of_difference = scipy.linalg.norm(cg_linear.x_k - x_star, ord=2)
        print(f"Exact solution = {x_star}")
        print(f"Norm of the difference between exact and found solutions = {norm_of_difference},")
        print(f" between starting point and exact solution = {scipy.linalg.norm(x_star, ord=2)}")
        print('**' * 15)



