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
    elif name == "SD":
        return newton_family.SteepestDescent(**kwargs)
    elif name == "SR1_TR":
        return quasi_newton_family.SR1_TR(**kwargs)


def print_distance_to_solution(x_found, solutions, norm):
    diffs = [np.linalg.norm(x_sol - x_found, ord=norm) for x_sol in solutions]
    closest_solution_index = np.argmin(diffs)
    difference = diffs[closest_solution_index]
    closest_solution = solutions[closest_solution_index]
    print(f"Closest exact solution is x_star = {closest_solution}")
    print(f"Distance(norm) from final iterate to the closest exact solution = {difference}")


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
    fs_names = ["RosenBrock: approximated",
                "RosenBrock: exact mode",
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

    print("#" * 50)
    print("Newton task: Standard vs Cholesky Modification")
    print("#" * 50)

    run(fs, fs_names, start_points, solutions, ["NM", "NM_Cholesky"], c=0.4, rho=0.5)

    print("#" * 50)
    print("Linear CG task")
    print("#" * 50)
    hilbert_fs = []
    hilbert_solutions = []
    hilbert_fs_names = []
    hilbert_starts = []

    for n in [5, 8, 12, 20, 30]:
        A = scipy.linalg.hilbert(n=n)
        b = np.ones(n, dtype=float)
        x_star = np.linalg.solve(A, b)
        hilbert_f = func.Quadratic(A=A, b=b)
        hilbert_fs.append(hilbert_f)
        hilbert_fs_names.append(f"HILBERT MATRIX {n} x {n}")
        hilbert_solutions.append([x_star])
        hilbert_starts.append([np.zeros(n, dtype=float)])

    run(hilbert_fs, hilbert_fs_names, hilbert_starts, hilbert_solutions, ["CG_Linear", "SD"], max_iterations=1e5)

    print("#" * 50)
    print("Nonlinear CG task: both F-R and P-R ")
    print("#" * 50)

    run(fs, fs_names, start_points, solutions, ["CG_FR", "CG_PR"])

    print("#" * 50)
    print("QN task: SR1 and BFGS")
    print("#" * 50)

    run(fs, fs_names, start_points, solutions, ["BFGS"], c=0.5, rho=0.9)
    run(fs, fs_names, start_points, solutions, ["SR1"], c=0, rho=0.99)

    print("#" * 50)
    print("QN bonus task: SR1 Trust Region")
    print("#" * 50)

    run(fs, fs_names, start_points, solutions, ["SR1_TR"])
