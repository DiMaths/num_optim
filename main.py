import numpy as np
import scipy.linalg

import func
import conjugate_gradient as cg
import newton_family

if __name__ == '__main__':

    fs = [func.RosenBrock(dim=2, num_mode=True),
          func.RosenBrock(dim=2, num_mode=False),
          func.SecondObjective(dim=2, num_mode=True),
          func.SecondObjective(dim=2, num_mode=False)]
    fs_names = ["Rosenblock: approximated",
                "Rosenblock: exact mode",
                "Alternative: approximated",
                "Alternative: exact mode"]
    points = [[np.array([1.2, 1.2]), np.array([-1.2, 1.]), np.array([0.2, 0.8])],
              [np.array([1.2, 1.2]), np.array([-1.2, 1.]), np.array([0.2, 0.8])],
              [np.array([-0.2, 1.2]), np.array([3.8, 0.1]), np.array([1.9, 0.6])],
              [np.array([-0.2, 1.2]), np.array([3.8, 0.1]), np.array([1.9, 0.6])]]

    print("#" * 50)
    print("Newton task: Standard vs Cholesky Modification")
    print("#" * 50)

    for i, f in enumerate(fs):
        print()
        print('|' * 50)
        print(f'Experiments on "{fs_names[i]}" function')
        print('|' * 50)
        for point in points[i]:
            print(f'***** STARTING POINT = {point} *****')
            test_alg = newton_family.NewtonMethod(f, point, c=0, rho=0.9)
            test_alg.execute()
            print()
            test_alg = newton_family.NewtonMethodModified(f, point, c=0, rho=0.9)
            test_alg.execute()
            print()

    print("#" * 50)
    print("Nonlinear CG task: both F-R and P-R ")
    print("#" * 50)

    cg_names = ["FR", "PR"]

    for i, f in enumerate(fs):
        print()
        print('|' * 50)
        print(f'Experiments on "{fs_names[i]}" function')
        print('|' * 50)
        for point in points[i]:
            print(f'***** STARTING POINT = {point} *****')
            for name in cg_names:
                test_alg = cg.NonlinearConjugateGradient(f, name, point)
                test_alg.execute()
                print()

    print("#"* 50)
    print("Linear CG task")
    print("#" * 50)

    for n in [5, 8, 12, 20, 30]:
        print('*' * 25)
        print(f"HILBERT MATRIX {n} x {n}")
        print('*' * 25)
        A = scipy.linalg.hilbert(n=n)
        x_star = np.linalg.solve(A, np.ones(n))

        hilbert_quadratic = func.Quadratic(A=A, b=np.ones(n))
        sd_hilbert = newton_family.SteepestDescent(hilbert_quadratic, max_iterations=10 ** 4)
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



