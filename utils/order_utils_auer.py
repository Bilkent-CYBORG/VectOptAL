import math

import numpy as np
import cvxpy as cp
import scipy as sp
import scipy.linalg


def m(i, j, i_v, j_v):
    output_dim = i.shape[0]

    res = np.inf
    for d in range(output_dim):
        yid = cp.Variable()
        yjd = cp.Variable()
        s = cp.Variable()

        cons1 = yid <= i[d] + i_v[d]
        cons2 = yid >= i[d] - i_v[d]
        cons3 = yjd <= j[d] + j_v[d]
        cons4 = yjd >= j[d] - j_v[d]
        cons5 = s >= 0
        cons6 = yid + s >= yjd

        objective = cp.Minimize(s)
        problem = cp.Problem(
            objective,
            [cons1, cons2, cons3, cons4, cons5, cons6]
        )
        problem.solve()

        if s.value <= res:
            res = s.value
        
    return max(0, res)


def M(i, j, i_v, j_v):
    yi1 = cp.Variable()
    yi2 = cp.Variable()
    yj1 = cp.Variable()
    yj2 = cp.Variable()
    s = cp.Variable()

    cons1 = yi1 <= i[0] + i_v[0]
    cons2 = yi1 >= i[0] - i_v[0]
    cons3 = yj1 <= j[0] + j_v[0]
    cons4 = yj1 >= j[0] - j_v[0]
    cons5 = yi2 <= i[1] + i_v[1]
    cons6 = yi2 >= i[1] - i_v[1]
    cons7 = yj2 <= j[1] + j_v[1]
    cons8 = yj2 >= j[1] - j_v[1]
    
    cons9 = s >= 0
    cons10 = yi1 <= yj1 + s
    cons11 = yi2 <= yj2 + s

    objective = cp.Minimize(s)
    problem = cp.Problem(
        objective,
        [cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons8, cons9, cons10, cons11]
    )
    problem.solve()

    return s.value


print(
    np.allclose(m(
        np.array([1,1]), np.array([3,3]),
        np.array([0.25,0.25]), np.array([0.25,0.25])
    ), 1.5)
)

print(
    np.allclose(m(
        np.array([1,1]), np.array([1.25,1.25]),
        np.array([0.5,0.5]), np.array([0.5,0.5])
    ), 0)
)

print(
    np.allclose(M(
        np.array([1,1]), np.array([3,3]),
        np.array([0.25,0.25]), np.array([0.25,0.25])
    ), 0)
)

print(
    M(
        np.array([1,1]), np.array([1.25,1.25]),
        np.array([0.5,0.5]), np.array([0.5,0.5])
    )
)
