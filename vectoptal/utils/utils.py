import itertools

import torch
import numpy as np
import cvxpy as cp
import scipy.special
from scipy.stats.qmc import Sobol
from sklearn.metrics.pairwise import euclidean_distances


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_2d_w(cone_angle):
    angle_radian = (cone_angle/180) * np.pi
    if cone_angle <= 90:
        W_1 = np.array([-np.tan(np.pi/4-angle_radian/2), 1])
        W_2 = np.array([+np.tan(np.pi/4+angle_radian/2), -1])
    else:
        W_1 = np.array([-np.tan(np.pi/4-angle_radian/2), 1])
        W_2 = np.array([-np.tan(np.pi/4+angle_radian/2), 1])
    W_1 = W_1/np.linalg.norm(W_1)
    W_2 = W_2/np.linalg.norm(W_2)
    W = np.vstack((W_1, W_2))

    return W

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

def get_closest_indices_from_points(
    pts_to_find, pts_to_check, return_distances=False, squared=False
):
    if len(pts_to_find) == 0 or len(pts_to_check) == 0:
        return []

    distances = euclidean_distances(pts_to_find, pts_to_check, squared=squared)
    x_inds = np.argmin(distances, axis=1)
    if return_distances:
        return x_inds.astype(int), np.min(distances, axis=1)
    return x_inds.astype(int)

def get_noisy_evaluations_chol(means, cholesky_cov):
    """Used for vectorized multivariate normal sampling."""
    n, d = means.shape[0], len(cholesky_cov)
    X = np.random.normal(size=(n, d))
    complicated_X = X.dot(cholesky_cov)

    noisy_samples = complicated_X + means
    
    return noisy_samples

def generate_sobol_samples(dim, n):
    sampler = Sobol(dim, scramble=False)
    samples = sampler.random(n)

    return samples

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
    # TODO: Check if the commented out early stop condition is correct.
    # if np.dot((vi-vj).T, vi-vj) <= eps**2:
    #     return True
    return is_covered_SOCP(vi, vj, eps, W)


def hyperrectangle_check_intersection(
    lower1: np.ndarray, upper1: np.ndarray, lower2: np.ndarray, upper2: np.ndarray
):
    if np.any(lower1 >= upper2) or np.any(upper1 <= lower2):
        return False
    
    return True

def hyperrectangle_get_vertices(lower: np.ndarray, upper: np.ndarray):
    a = [[l1, l2] for l1, l2 in zip(lower, upper)]
    vertex_list = [element for element in itertools.product(*a)]
    return np.array(vertex_list)

def hyperrectangle_get_region_matrix(lower: np.ndarray, upper: np.ndarray):
    dim = len(lower)
    region_matrix = np.vstack((np.eye(dim), -np.eye(dim)))
    region_boundary = np.hstack((np.array(lower), -np.array(upper)))

    return region_matrix, region_boundary

def is_pt_in_extended_polytope(pt, polytope, invert_extension=False):
    dim = polytope.shape[1]
    
    if invert_extension:
        comp_func = lambda x, y: x >= y
    else:
        comp_func = lambda x, y: x <= y

    # Vertex is trivially an element
    for vert in polytope:
        if comp_func(vert, pt).all():
            return True

    # Check intersections with polytope. If any intersection is dominated, then an element.
    for dim_i in range(dim):
        edges_of_interest = np.empty((0, 2, dim), dtype=np.float64)
        for vert_i, vert1 in enumerate(polytope):
            for vert_j, vert2 in enumerate(polytope):
                if vert_i == vert_j:
                    continue

                if vert1[dim_i] <= pt[dim_i] and pt[dim_i] <= vert2[dim_i]:
                    edges_of_interest = np.vstack((
                        edges_of_interest,
                        np.expand_dims(np.vstack((vert1, vert2)), axis=0)
                    ))

        for edge in edges_of_interest:
            intersection = line_seg_pt_intersect_at_dim(edge[0], edge[1], pt, dim_i)
            if intersection is not None and comp_func(intersection, pt).all():
                # Vertex is an element due to the intersection point
                return True

    return False

def line_seg_pt_intersect_at_dim(P1, P2, target_pt, target_dim):
    t = (target_pt[target_dim] - P1[target_dim]) / (P2[target_dim] - P1[target_dim])

    if t < 0 or t > 1:
        # No intersection
        return None

    point_on_line = P1 + t * (P2 - P1)
    return point_on_line

def binary_entropy(x):
    return -(scipy.special.xlogy(x, x) + scipy.special.xlog1py(1 - x, -x)) / np.log(2)
