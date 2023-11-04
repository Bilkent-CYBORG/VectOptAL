import numpy as np
import cvxpy as cp


def print_list(list):
    print("List length: ", len(list))
    for x in list:
        print(x)

# Helper functions

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
    h = -np.array([np.max([0,np.dot(W[0,:],vj-vi)[0]]),
                np.max([0,np.dot(W[1,:],vj-vi)[0]])])

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                 [G @ x <= h])
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
    prob.solve()

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
    prob.solve()

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
    #if np.dot((vi-vj).T, vi-vj) <= eps**2:
    #    return True
    return is_covered_SOCP(vi, vj, eps, W)

    
def get_pareto_set(mu, W, alpha_vec, return_mask = False):
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
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient 

    
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
        uncovered = True
        for j in p_opt_hat:
            if is_covered(mu[i,:].reshape(-1,1), mu[j,:].reshape(-1,1), eps, W):
                uncovered = False
                break
        
        if uncovered:
            uncovered_set.append(i)
        
    return np.array(uncovered_set)
