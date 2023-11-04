import math

import numpy as np
import cvxpy as cp

from utils.utils import get_smallmij


def x_not_dominated_by_y(W, mx, my, r_x, r_y, alpha_vec, epsilon):
    output_dim = mx.shape[0]
    
    mux = cp.Variable(output_dim)  # mux and muy are from ellipses of mx and my
    muy = cp.Variable(output_dim)
    
    cons1 = cp.norm(mux - mx) <= r_x
    cons2 = cp.norm(muy - my) <= r_y

    cons3 = W @ (muy - mux) >= epsilon * alpha_vec[:, 0]

    conss = [cons1, cons2, cons3]
    
    objective = cp.Minimize(0)
    
    prob = cp.Problem(objective, conss)
    
    try:
        prob.solve()  # "GUROBI", "MOSEK"
    except:
        prob.solve(solver="GUROBI")
    
    if "optimal" in prob.status:  # "optimal" or "optimal_inaccurate"
        return False
    else:
        return True

def is_x_eps_pareto(A_non_discarded, design_x, W, means, variances, alpha_vec, epsilon):
    x_i = design_x[-1].astype(int)

    for design_y in A_non_discarded:
        if np.array_equal(design_x[:-1], design_y[:-1]):
            continue

        y_i = design_y[-1].astype(int)
        
        if not x_not_dominated_by_y(
            W, means[x_i], means[y_i], variances[x_i], variances[y_i], alpha_vec, epsilon
        ):
            return False
    
    return True


def sure_y_dominates_x(W, mx, my, r_x, r_y, alpha_vec, epsilon):
    output_dim = mx.shape[0]
    mux = cp.Variable(output_dim)
    muy = cp.Variable(output_dim)
    
    cons1 = cp.norm(mux - mx) <= r_x
    cons2 = cp.norm(muy - my) <= r_y
    
    conss = [cons1, cons2]

    for n in range(W.shape[0]):
        objective = cp.Minimize(W[n] @ (muy - mux))
        
        prob = cp.Problem(objective, conss)
        try:
            prob.solve()  # "GUROBI", "MOSEK"
        except:
            prob.solve(solver="GUROBI")

        if prob.value < 0:  # -alpha_vec[n] * epsilon:
            return False
    
    return True

def is_x_dominated(A_non_discarded, design_x, W, means, variances, alpha_vec, epsilon):
    x_i = design_x[-1].astype(int)

    for design_y in A_non_discarded:
        if np.array_equal(design_x[:-1], design_y[:-1]):
            continue

        y_i = design_y[-1].astype(int)
        
        if sure_y_dominates_x(
            W, means[x_i], means[y_i], variances[x_i], variances[y_i], alpha_vec, epsilon
        ):
            return True
    
    return False
