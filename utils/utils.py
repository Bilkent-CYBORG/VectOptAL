import os
import shutil
import pickle
from pathlib import Path
from fractions import Fraction

import torch
import numpy as np
import cvxpy as cp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True


def read_sorted_results(alg_path, sort=True):
    result_file_path = os.path.join(alg_path, "results.pkl")
    with open(result_file_path, "rb") as results_file:
        results_list = pickle.load(results_file)
    
    if sort:
        # results_list.sort(key=lambda x:x["batch_size"])
        results_list.sort(key=lambda x:x["conf_contraction"])
        results_list.sort(key=lambda x:x["eps"], reverse=True)
        results_list.sort(key=lambda x:x["cone_degree"])
        results_list.sort(key=lambda x:x["dataset_name"])

    return results_list


def overwrite_makedirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def incremental_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path

    tmp_path = path
    i = 1
    while os.path.exists(tmp_path):
        tmp_path = f"{path}_{i}"
        i += 1
    
    os.makedirs(tmp_path)
    return tmp_path


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def save_new_result(output_folder_path, experiment_res_dict):
    res_pickle_path = os.path.join(output_folder_path, "results.pkl")
    res_pickle_tmp_path = os.path.join(output_folder_path, "results_tmp.pkl")

                # Create a result file if it doesn't exist
    if not os.path.exists(res_pickle_path):
        with open(res_pickle_path, "wb") as results_file:
            pickle.dump([], results_file)
                # Load the results until now
    with open(res_pickle_path, "rb") as results_file:
        results_list = pickle.load(results_file)
                # Add new experiment results
    results_list.append(experiment_res_dict)
                # Rename current result file as tmp in case of an error
    os.rename(res_pickle_path, res_pickle_tmp_path)
                # Write new results
    with open(res_pickle_path, "wb") as results_file:
        pickle.dump(results_list, results_file)
                # Remove the tmp one if the new one is successfuly written
    os.remove(res_pickle_tmp_path)


def get_noisy_evaluations_chol(means, cholesky_cov):
    """Used for vectorized multivariate normal sampling."""
    n, d = means.shape
    X = np.random.normal(size=(n, d))
    
    noisy_samples = X.dot(cholesky_cov) + means
    
    return noisy_samples


def get_cone_text(degree):
    fraction = Fraction(degree, 180).limit_denominator()
    numerator, denominator = fraction.numerator, fraction.denominator

    if numerator == 1:
        return rf'$C_\theta=\pi / {denominator}$'
    else:
        return rf'$C_\theta={numerator}\pi / {denominator}$'


def get_cone_params(degree, dim=2):
    if degree == 45:
        theta_45 = np.pi/4
        W_45_1 = np.array([-np.tan(np.pi/4-theta_45/2), 1])
        W_45_2 = np.array([+np.tan(np.pi/4+theta_45/2), -1])
        W_45_1 = W_45_1/np.linalg.norm(W_45_1)
        W_45_2 = W_45_2/np.linalg.norm(W_45_2)
        W = np.vstack((W_45_1, W_45_2))
        alpha_vec = np.array([[0.70710678], [0.70710678]])
    elif degree == 60:
        theta_60 = np.pi/3
        W_60_1 = np.array([-np.tan(np.pi/4-theta_60/2), 1])
        W_60_2 = np.array([+np.tan(np.pi/4+theta_60/2), -1])
        W_60_1 = W_60_1/np.linalg.norm(W_60_1)
        W_60_2 = W_60_2/np.linalg.norm(W_60_2)
        W = np.vstack((W_60_1, W_60_2))
        alpha_vec = np.array([[0.8660254], [0.8660254]])
    elif degree == 90:
        W = np.eye(dim)
        alpha_vec = np.ones((dim, 1))
    elif degree == 120:
        theta_120 = 2*np.pi/3
        W_120_1 = np.array([-np.tan(np.pi/4-theta_120/2), 1])
        W_120_2 = np.array([-np.tan(np.pi/4+theta_120/2), 1])
        W_120_1 = W_120_1/np.linalg.norm(W_120_1)
        W_120_2 = W_120_2/np.linalg.norm(W_120_2)
        W = np.vstack((W_120_1, W_120_2))
        alpha_vec = np.array([[1], [1]])
    elif degree == 135:
        theta_135 = 3*np.pi/4
        W_135_1 = np.array([-np.tan(np.pi/4-theta_135/2), 1])
        W_135_2 = np.array([-np.tan(np.pi/4+theta_135/2), 1])
        W_135_1 = W_135_1/np.linalg.norm(W_135_1)
        W_135_2 = W_135_2/np.linalg.norm(W_135_2)
        W = np.vstack((W_135_1, W_135_2))
        alpha_vec = np.array([[1], [1]])
    else:
        raise ValueError

    return W, alpha_vec, get_cone_text(degree)


def get_alpha(rind, W):
    """
    Compute alpha_rind for row rind of W 
    :param rind: row index
    :param W: (n_constraint,D) ndarray
    :return: alpha_rind.
    """
    m = W.shape[0]+1 #number of constraints
    D = W.shape[1]
    f = -W[rind,:]
    A = []
    b = []
    c = []
    d = []
    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.zeros(1))
    
    A.append(np.eye(D))
    b.append(np.zeros(D))
    c.append(np.zeros(D))
    d.append(np.ones(1))

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
          cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T@x),
                  soc_constraints)
    prob.solve(solver="ECOS")

    """
    # Print result.
    print("The optimal value is", -prob.value)
    print("A solution x is")
    print(x.value)
    for i in range(m):
        print("SOC constraint %i dual variable solution" % i)
        print(soc_constraints[i].dual_value)
    """    
        
    return -prob.value   


def get_alpha_vec(W):
    """
    Compute alpha_vec for W 
    :param W: an (n_constraint,D) ndarray
    :return: alpha_vec, an (n_constraint,1) ndarray
    """    
    alpha_vec = np.zeros((W.shape[0],1))
    for i in range(W.shape[0]):
        alpha_vec[i] = get_alpha(i, W)
    return alpha_vec


def get_bigmij(vi, vj, W):
    """
    Compute M(i,j) for designs i and j 
    :param vi, vj: (D,1) ndarrays
    :param W: (n_constraint,D) ndarray
    :return: M(i,j).
    """
    D = W.shape[1]
    P = 2*np.eye(D)
    q = (-2*(vj-vi)).ravel()
    G = -W
    h = -np.array([np.max([0,np.dot(row, vj-vi)[0]]) for row in W])
    # h = -np.array([
    #     np.max([0,np.dot(W[0,:], vj-vi)[0]]),
    #     np.max([0,np.dot(W[1,:], vj-vi)[0]])
    # ])

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    prob = cp.Problem(
        cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
        [G @ x <= h]
        )
    #A @ x == b    
    prob.solve()
    bigmij = np.sqrt(prob.value + np.dot((vj-vi).T, vj-vi)).ravel()

    # Print result.
    #print("\nThe optimal value is", prob.value)
    #print("A solution x is")
    #print(x.value)
    #print("A dual solution corresponding to the inequality constraints is")
    #print(prob.constraints[0].dual_value)
    #print("M(i,j) is", bigmij)
    return bigmij


def get_smallmij(vi, vj, W, alpha_vec):
    """
    Compute m(i,j) for designs i and j 
    :param vi, vj: (D,1) ndarrays
    :param W: (n_constraint,D) ndarray
    :param alpha_vec: (n_constraint,1) ndarray of alphas of W
    :return: m(i,j).
    """
    prod = np.matmul(W, vj - vi)
    prod[prod<0] = 0
    smallmij = (prod/alpha_vec).min()
    
    return smallmij


def get_delta(mu, W, alpha_vec):
    """
    Computes Delta^*_i for each i in [n.points]
    :param mu: An (n_points, D) array
    :param W: (n_constraint,D) ndarray
    :param alpha_vec: (n_constraint,1) ndarray of alphas of W
    :return: An (n_points, D) array of Delta^*_i for each i in [n.points]
    """
    n = mu.shape[0]
    Delta = np.zeros(n)
    for i in range(n):
        for j in range(n):
            vi = mu[i,:].reshape(-1,1)
            vj = mu[j,:].reshape(-1,1)
            mij = get_smallmij(vi, vj, W, alpha_vec)
            if mij>Delta[i]:
                Delta[i] = mij
    
    return Delta.reshape(-1,1)


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def get_uncovered_set(p_opt_miss, p_opt_hat, mu, eps, W):
    """
    Check if vi is eps covered by vj for cone matrix W
    :param p_opt_hat: ndarray of indices of designs in returned Pareto set
    :param p_opt_miss: ndarray of indices of Pareto optimal points not in p_opt_hat
    :mu: An (n_points,D) mean reward matrix
    :param eps: float
    :param W: An (n_constraint,D) ndarray
    :return: ndarray of indices of points in p_opt_miss that are not epsilon covered
    """
    uncovered_set = []
    
    for i in p_opt_miss:
        for j in p_opt_hat:
            if is_covered(mu[i,:].reshape(-1,1), mu[j,:].reshape(-1,1), eps, W):
                break
        else:
            uncovered_set.append(i)
        
    return uncovered_set


def is_covered_SOCP(vi, vj, eps, W):
    """
    Check if vi is eps covered by vj for cone matrix W 
    :param vi, vj: (D,1) ndarrays
    :param W: An (n_constraint,D) ndarray
    :param eps: float
    :return: Boolean.
    """    
    m = 2*W.shape[0]+1 # number of constraints
    D = W.shape[1]
    f = np.zeros(D)
    A = []
    b = []
    c = []
    d = []

    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.zeros(1))
    
    A.append(np.eye(D))
    b.append((vi-vj).ravel())
    c.append(np.zeros(D))
    d.append(eps*np.ones(1))

    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.dot(W[i,:],(vi-vj)))
        
    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
          cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T@x),
                  soc_constraints)
    prob.solve(solver="ECOS")

    """
    # Print result.
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print(x.value is not None)
    for i in range(m):
        print("SOC constraint %i dual variable solution" % i)
        print(soc_constraints[i].dual_value)
    """     
    return x.value is not None


def is_covered(vi, vj, eps, W):
    """
    Check if vi is eps covered by vj for cone matrix W 
    :param vi, vj: (D,1) ndarrays
    :param W: An (n_constraint,D) ndarray
    :param eps: float
    :return: Boolean.
    """
    if np.dot((vi-vj).T, vi-vj) <= eps**2:
        return True
    return is_covered_SOCP(vi, vj, eps, W)


def get_pareto_set(mu, W, alpha_vec):
    """
    Find the indices of Pareto designs (rows of mu)
    :param mu: An (n_points, D) array
    :param W: (n_constraint,D) ndarray
    :param alpha_vec: (n_constraint,1) ndarray of alphas of W
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(mu.shape[0])
    n_points = mu.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(mu):
        nondominated_point_mask = np.zeros(mu.shape[0], dtype=bool)
        vj = mu[next_point_index].reshape(-1,1)
        for i in range(len(mu)):
            vi = mu[i].reshape(-1,1)
            nondominated_point_mask[i] =  (get_smallmij(vi, vj, W, alpha_vec) == 0) and (get_bigmij(vi, vj, W) > 0)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        mu = mu[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    return is_efficient


def plot_pareto_front(
    func_val1=None, func_val2=None,
    mask=None,
    y1=None, y2=None,
    plotfront=False,
    title=r'$Objective Space$', f1label=r'$f^1$', f2label=r'$f^2$',
    save_path=None
):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes()

    if func_val1 is not None:
        ax.scatter(
            func_val1[mask==False], func_val2[mask==False],
            color='gray', alpha=0.5,  # label='Pareto olmayan',
        )
        ax.scatter(
            func_val1[mask], func_val2[mask],
            color='tab:green', label='Gerçek Pareto', alpha=0.7
        )

    if y1 is not None:
        ax.scatter(
            y1, y2,
            s=16, color='tab:red', marker='x', label="Tahmini Pareto", alpha=0.9
        )

    ax.set_xlabel(f1label)
    ax.set_ylabel(f2label)
    plt.title(title)
    plt.grid(True, linewidth=0.5, color='gray', linestyle='-')

    plt.legend()

    if save_path is not None:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
