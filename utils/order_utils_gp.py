import math

import numpy as np
import cvxpy as cp
import scipy as sp

from utils.utils import get_smallmij


def x_not_dominated_by_y(W, mx, my, sigma_x, sigma_y, alpha, alpha_vec, epsilon):
    output_dim = mx.shape[0]
    mux = cp.Variable(output_dim)  # mux and muy are from ellipses of mx and my
    muy = cp.Variable(output_dim)

    # # norm( Qsqrt * ( A * x - b ) ) <= 1
    cons1 = cp.norm(sp.linalg.sqrtm(np.linalg.inv(sigma_x)) @ (mux - mx).T) <= alpha
    cons2 = cp.norm(sp.linalg.sqrtm(np.linalg.inv(sigma_y)) @ (muy - my).T) <= alpha
    
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

def is_x_eps_pareto(A_non_discarded, design_x, W, means, variances, alpha, alpha_vec, epsilon):
    x_i = design_x[-1].astype(int)

    for design_y in A_non_discarded:
        if np.array_equal(design_x[:-1], design_y[:-1]):
            continue

        y_i = design_y[-1].astype(int)
        
        if not x_not_dominated_by_y(
            W, means[x_i], means[y_i], variances[x_i], variances[y_i], alpha, alpha_vec, epsilon
        ):
            return False
    
    return True


def sure_y_dominates_x(W, mx, my, sigma_x, sigma_y, alpha, alpha_vec, epsilon):
    output_dim = mx.shape[0]
    mux = cp.Variable(output_dim)
    muy = cp.Variable(output_dim)
    
    # # quad_form( A * x - b, Q ) <= 1
    # cons1 = cp.quad_form((mux - mx).T, np.linalg.inv(sigma_x)) <= alpha
    # cons2 = cp.quad_form((muy - my).T, np.linalg.inv(sigma_y)) <= alpha
    # # norm( Qsqrt * ( A * x - b ) ) <= 1
    cons1 = cp.norm(sp.linalg.sqrtm(np.linalg.inv(sigma_x)) @ (mux - mx).T) <= alpha
    cons2 = cp.norm(sp.linalg.sqrtm(np.linalg.inv(sigma_y)) @ (muy - my).T) <= alpha
    
    conss = [cons1, cons2]

    for n in range(W.shape[0]):
        objective = cp.Minimize(W[n] @ (muy - mux))
        
        prob = cp.Problem(objective, conss)
        try:
            prob.solve()  # "GUROBI", "MOSEK"
        except:
            prob.solve(solver="GUROBI")

        if prob.value < -alpha_vec[n] * epsilon:
            return False
    
    return True

def is_x_dominated(A_non_discarded, design_x, W, means, variances, alpha, alpha_vec, epsilon):
    x_i = design_x[-1].astype(int)

    for design_y in A_non_discarded:
        if np.array_equal(design_x[:-1], design_y[:-1]):
            continue

        y_i = design_y[-1].astype(int)
        
        if sure_y_dominates_x(
            W, means[x_i], means[y_i], variances[x_i], variances[y_i], alpha, alpha_vec, epsilon
        ):
            return True
    
    return False


### MULTI OBJECTIVE RECTANGLE

def is_x_dominated_rect(
    A_non_discarded, design_x, W, means, variances, alpha, alpha_vec, epsilon
):
    x_i = design_x[-1].astype(int)
    bx = np.array(means[x_i] + alpha*np.sqrt(np.diag(variances[x_i])))

    for design_y in A_non_discarded:
        if np.array_equal(design_x[:-1], design_y[:-1]):
            continue

        y_i = design_y[-1].astype(int)
        by = np.array(means[y_i] - alpha*np.sqrt(np.diag(variances[y_i])))

        if np.all(by - bx >= -epsilon):
            return True
    
    return False

def is_x_eps_pareto_rect(
    A_non_discarded, design_x, W, means, variances, alpha, alpha_vec, epsilon
):
    x_i = design_x[-1].astype(int)
    bx = np.array(means[x_i] - alpha*np.sqrt(np.diag(variances[x_i])))
    
    for design_y in A_non_discarded:
        if np.array_equal(design_x[:-1], design_y[:-1]):
            continue
        
        y_i = design_y[-1].astype(int)
        by = np.array(means[y_i] + alpha*np.sqrt(np.diag(variances[y_i])))

        if np.min(by - bx) > epsilon:
            return False

    return True
