import numpy as np
import scipy.linalg

import func
import conjugate_gradient as cg
import newton_family
import quasi_newton_family


def create_alg(name: str, **kwargs):
    if name == "CG_Linear":
        return cg.LinearConjugateGradient(**kwargs)
    elif name.startswith("CG_"):
        return cg.NonlinearConjugateGradient(**kwargs, name=name)
    elif name == "SR1":
        return quasi_newton_family.SR1(**kwargs)
    elif name == "BFGS":
        return quasi_newton_family.BFGS(**kwargs)
    elif name == "NM":
        return newton_family.NewtonMethod(**kwargs)
    elif name == "NM_Cholesky":
        return newton_family.NewtonMethodModified(**kwargs)


def print_distance_to_solution(x_found, solutions, norm, allowed_difference: float = 0.1):
    difference = np.min([np.linalg.norm(x_sol - x_found, ord=norm) for x_sol in solutions])
    print(f"Distance(norm) from final iterate to the closest exact solution = {difference}")
    if difference > allowed_difference:
        print(f"SOLUTION IS NOT FOUND AT ALL")


def run(fs, fs_names, start_points, solutions, algo_names, **kwargs):
    for i, f in enumerate(fs):
        print()
        print('|' * 50)
        print(f'Experiments on "{fs_names[i]}" function')
        print('|' * 50)
        for point in start_points[i]:
            print(f'{"*"*10}  STARTING POINT = {point}  {"*"*10}')
            for name in algo_names:
                test_alg = create_alg(name=name, f=f, start_point=point, **kwargs)
                test_alg.execute()
                print_distance_to_solution(test_alg.x_k, solutions[i], test_alg.norm)
                print()


if __name__ == '__main__':
    fs = [func.RosenBrock(dim=2, num_mode=True),
          func.RosenBrock(dim=2, num_mode=False),
          func.SecondObjective(dim=2, num_mode=True),
          func.SecondObjective(dim=2, num_mode=False)]
    fs_names = ["Rosenblock: approximated",
                "Rosenblock: exact mode",
                "Alternative: approximated",
                "Alternative: exact mode"]
    start_points = [[np.array([1.2, 1.2]), np.array([-1.2, 1.]), np.array([0.2, 0.8])],
                    [np.array([1.2, 1.2]), np.array([-1.2, 1.]), np.array([0.2, 0.8])],
                    [np.array([-0.2, 1.2]), np.array([3.8, 0.1]), np.array([1.9, 0.6])],
                    [np.array([-0.2, 1.2]), np.array([3.8, 0.1]), np.array([1.9, 0.6])]]
    solutions = [[np.array([1., 1.])],
                 [np.array([1., 1.])],
                 [np.array([0., 1.]), np.array([4., 0.])],
                 [np.array([0., 1.]), np.array([4., 0.])]]

    # run(fs, fs_names, start_points, solutions, ["SR1", "BFGS"], c=0., rho=0.975)

    print("#" * 50)
    print("Newton task: Standard vs Cholesky Modification")
    print("#" * 50)

    run(fs, fs_names, start_points, solutions, ["NM", "NM_Cholesky"], c=0., rho=0.9)

    print("#" * 50)
    print("Nonlinear CG task: both F-R and P-R ")
    print("#" * 50)

    run(fs, fs_names, start_points, solutions, ["CG_FR", "CG_PR"])

    """print("#"* 50)
        print("Linear CG task")
        print("#" * 50)

        for n in [3, 4, 5, 6, 7, 8]:
            print('*' * 25)
            print(f"HILBERT MATRIX {n} x {n}")
            print('*' * 25)
            A = scipy.linalg.hilbert(n=n)
            b = np.ones(n, dtype=float)
            x_star = np.linalg.solve(A, b)

            hilbert_quadratic = func.Quadratic(A=A, b=b)
            sd_hilbert = newton_family.SteepestDescent(hilbert_quadratic, max_iterations=10 ** 4)
            sd_hilbert.execute()

            norm_of_difference = np.linalg.norm(sd_hilbert.x_k - x_star, ord=2)
            print(f"Distance(norm) from final iterate to the closest exact solution = {norm_of_difference}")
            print('**' * 15)

            cg_linear = cg.LinearConjugateGradient(hilbert_quadratic)
            cg_linear.execute()

            norm_of_difference = np.linalg.norm(cg_linear.x_k - x_star, ord=2)
            print(f"Distance(norm) from final iterate to the closest exact solution = {norm_of_difference}")
            print('**' * 15)"""
