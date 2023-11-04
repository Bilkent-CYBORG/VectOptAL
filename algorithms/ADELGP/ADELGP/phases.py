import cvxpy as cp
import numpy as np
import copy
from typing import List

from algorithms.ADELGP.ADELGP.utils import *
import scipy.stats as stat

from itertools import repeat
from concurrent.futures import ProcessPoolExecutor as Pool

from algorithms.ADELGP.ADELGP.Hyperrectangle import Hyperrectangle
from algorithms.ADELGP.ADELGP.Polyhedron import Polyhedron
from algorithms.ADELGP.ADELGP.DesignPoint import DesignPoint
import torch


def modeling(A, gp, beta, C ,t):
    
    xs = torch.tensor([[list(point.x) for point in A]])
   
    mu, sigma, cov = gp.inference_bulk(xs)
    cov_list = [torch.tensor([cov[0,i,i].cpu().item(),cov[0,i+1,i+1].cpu().item()]) for i in range(0,cov.shape[1],2)]
    mean_tensor = mu.reshape(-1,2)
    for ind,point in enumerate(A):
        point.update_cumulative_conf_rect(mean_tensor[ind].reshape(-1,1), torch.concat((cov_list[ind].reshape(-1,1),cov_list[ind].reshape(-1,1)),axis=1), beta, t)
 

    """ for ind,point in enumerate(A):
        point.update_cumulative_conf_rect(mu, cov, beta, t)
        mu, sigma, cov = gp.inference(point.x)
        point.update_cumulative_conf_rect(mu, cov, beta, t) """

    """ for point in A:
        mu, sigma, cov = gp.inference(point.x)
        point.update_cumulative_conf_rect(mu, cov, beta, t) """


def discard(S, P, C, epsilon):
    A = S + P
    p_pess = pess(A, C)
    difference = set_diff(S, p_pess) #undecided points that are not in pessimistic pareto set.
    for point in difference:
        for point_prime in p_pess:
            # Function to check if  ∃z' in R(x') such that R(x) <_C z + u, where u < epsilon
            # print("discard step")
            # print(point, point_prime)
            if dominated_by_opt3(point, point_prime, C, epsilon):
                S.remove(point)
                break
    """ for point in S:
        for point_prime in A:
            if point == point_prime:
                continue
            if dominated_by_opt3(point, point_prime, C, epsilon):
                S.remove(point)
                break  """

# def epsilon_covering_parallel(S, A, C, epsilon, S_fst_half_flag):
#     l = len(S)

#     if S_fst_half_flag:
#         S_tmp = S[:l//2]
#     else:
#         S_tmp = S[l//2:]

#     results = []

#     for point in S_tmp:
#         for point_prime in A:
#             if point == point_prime:
#                 continue
#             # Function to check if  ∃x' in W_t  for epsilon-covering condition
#             # print(point, point_prime)

#             if ecovered_faster(point, point_prime, C, epsilon):
#                 results.append(True)
#                 break
        
#             results.append(False)

#     return results

# def epsiloncovering(S, P, C, epsilon):
#     A = S + P

#     with Pool(max_workers=2) as pool:
#         results = pool.map(
#             epsilon_covering_parallel,
#             repeat(S),
#             repeat(A),
#             repeat(C),
#             repeat(epsilon),
#             [True, False]
#         )
#     results = list(results)
#     results = results[0] + results[1]
    
#     true_indices = np.nonzero(results)
#     S_np = np.array(S)

#     P += list(S_np[np.where(results == False)[0]])
#     S = list(S_np[true_indices])
    
#     return None

def epsiloncovering(S, P, C, epsilon):
    A = S + P
    for point in S:
        exists = False
        for point_prime in A:
            if point == point_prime:
                continue
            # Function to check if  ∃x' in W_t  for epsilon-covering condition
            # print(point, point_prime)


            if ecovered_faster(point, point_prime, C, epsilon):
                exists = True
                break
        # If the loop does not break, there is an x' in W_t
        if not exists:
            S.remove(point)
            P.append(point)

    return None

def evaluate(W: List[DesignPoint], model,t,beta,cone,batched) -> DesignPoint:


    if batched:

        observe_list = list()
        batch_size = batched
        sample_cnt = min(len(W),batch_size)
        # ------ SUM OF VARIANCE CHOOSING
        tmp_mod = copy.deepcopy(model)

        # Choose maximum sum of variances
        for batch_i in range(sample_cnt):
            largest = 0
            to_observe = None
            for x in W:
                diameter = x.R.diameter
                if diameter > largest:
                    largest = diameter
                    to_observe = x
            observe_list.append(to_observe)
            tmp_mod.update(to_observe.x, torch.zeros(model.m))
            modeling(W, tmp_mod, beta, cone, t)
        return observe_list

    else:
        
        """ stat.norm().pdf() """

        
        """ if t<20:
            largest = 0
            to_observe = None
            for x in W:
                diameter = x.R.diameter
                if diameter > largest:
                    largest = diameter
                    to_observe = x
            print(f"Observing point {to_observe}. It has diameter {largest}")
        else:
            mus = np.array([designpoint.mu.cpu().numpy() for designpoint in W])
            #distances = [min([np.linalg.norm(mu-mu_prime) for mu_prime in np.delete(mus,index,0)]) for index,mu in enumerate(mus)]
            distances = [min([min(abs(mu-mu_prime).reshape(-1,)) for mu_prime in np.delete(mus,index,0)]) for index,mu in enumerate(mus)]

            #print(Delta)

            largest = -np.inf
            to_observe = None
            for ind,x in enumerate(W):
                diameter = x.R.diameter
                dist = distances[ind]
                if diameter/(dist/(np.sum(distances))) > largest:
                    largest = diameter/(dist/(np.sum(distances)))
                    to_observe = x 
            print(f"{largest},{dist},{diameter},{to_observe},{to_observe.R.upper},{to_observe.R.lower}")
            #print(f"{diameter},{to_observe.R.upper},{to_observe.R.lower}") """

        """ if t==0:
            largest = 0
            to_observe = None
            for x in W:
                diameter = x.R.diameter
                if diameter > largest:
                    largest = diameter
                    to_observe = x
            print(f"Observing point {to_observe}. It has diameter {largest}")
        else:
            mus = np.array([designpoint.mu.cpu().numpy() for designpoint in W])
            alpha_vec = get_alpha_vec(cone.A)
            Delta = get_delta(mus, cone.A, alpha_vec)

            #print(Delta)

            largest = 0
            to_observe = None
            for ind,x in enumerate(W):
                diameter = x.R.diameter
                delta_norm = np.linalg.norm(Delta[ind])
                if diameter/(delta_norm+0.01) > largest:
                    largest = diameter/(delta_norm+0.01)
                    to_observe = x 
            print(f"{largest},{delta_norm},{diameter}") """
        



        largest = 0
        to_observe = None
        for x in W:
            diameter = x.R.diameter
            if diameter > largest:
                largest = diameter
                to_observe = x

        print(f"Observing point {to_observe}. It has diameter {largest}")
        
        return [to_observe]

""" 
def parallel_pess_innerloop(i,length,point_set,C):
    for j in range(length):
        if j == i:
            continue

        # Check if there is another point j that dominates i, if so, do not include i in the pessimistic set
        if check_dominates_faster(point_set[j].R, point_set[i].R, C):
            return False
    return True
 """


def pess(point_set: List[DesignPoint], C: Polyhedron) -> List[DesignPoint]:
    """
    The set of Pessimistic Pareto set of a set of DesignPoint objects.
    :param point_set: List of DesignPoint objects.
    :param C: The ordering cone.
    :return: List of Node objects.
    """
    pess_set = []
    length = len(point_set)

    for i in range(length):
        set_include = True

        for j in range(length):
            if j == i:
                continue

            # Check if there is another point j that dominates i, if so, do not include i in the pessimistic set
            if check_dominates(point_set[j].R, point_set[i].R, C):#check_dominates_faster(point_set[j].R, point_set[i].R, C):
                set_include = False
                break

        if set_include:
            pess_set.append(point_set[i])

    return pess_set

    """ length = len(point_set)

    with Pool(processes=4) as pool:
        results = pool.map(
            parallel_pess_innerloop,
            zip(
                range(length),
                repeat(length),
                repeat(point_set),
                repeat(C),
            )
        )

    pess_set = [point_set[i] for i, res in enumerate(results) if res]

    return pess_set """


def check_dominates(polyhedron1: Hyperrectangle, polyhedron2: Hyperrectangle, cone: Polyhedron) -> bool:
    """
    Check if polyhedron1 dominates polyhedron2.
    Check if polyhedron1 is a subset of polyhedron2 + cone (by checking each vertex of polyhedron1).

    :param polyhedron1: The first polyhedron.
    :param polyhedron2: The second polyhedron.
    :param cone: The ordering cone.
    :return: Dominating condition.
    """ 
            
    condition = True
    n = cone.A.shape[1]  # Variable shape of x
    c = np.zeros(n)   

    vertices = polyhedron1.get_vertices()



    for vertex in vertices:
        """
        Checking if vertices can be represented by summation of a help vector from cone (y in this case) and zx
        """
        x = cp.Variable(n)
        y = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c.T @ x),
                          [x + y == vertex, #@: Together with two lines below, this enforces vertextes of first
                                            # hyperrec to be sum of help from cone (y) plus point from polyhedron2.
                           polyhedron2.A @ x >= polyhedron2.b, #@: This enforces x to be in polyhedron2
                           cone.A @ y >= cone.b]) #@: This enforces y to be in cone
        try:                   
            prob.solve(solver = "ECOS")
        except cp.error.SolverError:
            prob.solve(solver = "SCIPY")

        if prob.status == 'infeasible':
            condition = False
            break


    return condition



def check_dominates_faster(polyhedron1: Hyperrectangle, polyhedron2: Hyperrectangle, cone: Polyhedron) -> bool:
    """
    Check if polyhedron1 dominates polyhedron2.
    Check if polyhedron1 is a subset of polyhedron2 + cone (by checking each vertex of polyhedron1).

    :param polyhedron1: The first polyhedron.
    :param polyhedron2: The second polyhedron.
    :param cone: The ordering cone.
    :return: Dominating condition.
    """ 
            
    condition = True
    n = cone.A.shape[1]  # Variable shape of x
    c = np.zeros(n)   

    vertices = polyhedron1.get_vertices()
    vertices_prime = polyhedron2.get_vertices()


    
    for vertex in vertices:
        for vertex_prime in vertices_prime:
            condition = (cone.A @ (vertex-vertex_prime) >= cone.b).all() and condition

    return condition




def set_diff(s1, s2):  # Discarding
    #@: implements s1-s2  where the shared subsets are removed from s1.
    """
    Set difference of two sets.

    :param s1: List of DesignPoint objects.
    :param s2: List of DesignPoint objects.
    :return: List of DesignPoint objects.
    """

    tmp = copy.deepcopy(s1)

    for node in s2:
        if node in tmp:
            tmp.remove(node)

    return tmp




def dominated_by_opt3(point, point_prime, C, epsilon): #Line 11 of the algorithm
    #implementing the discarding rule
    # Define and solve the CVXPY problem.

    n = C.A.shape[1]
    #u = cp.Variable(n)
    u = np.array([epsilon/np.sqrt(2),epsilon/np.sqrt(2)])
    W_C = C.A
    b_C = C.b
    # Check each vertex in R(x)
    condition = True
    vertices = point.R.get_vertices()
    vertices_prime = point_prime.R.get_vertices()


    boolean_cum = True
    for row in vertices:
        for row_prime in vertices_prime:
            z = row
            boolean_cum = (W_C @ z <= W_C @ (row_prime + u)).all() and boolean_cum
    return boolean_cum 
        






def ecovered(point, point_prime, C, epsilon): 
    """

    :param point: DesignPoint x.
    :param point_prime: Design Point x'.
    :param C: Polyhedron C.
    :param epsilon:
    :return:
    """
    n = C.A.shape[1]

    z = cp.Variable(n)
    z_point = cp.Variable(n)
    z_point2 = cp.Variable(n)
    c_point = cp.Variable(n)
    c_point2 = cp.Variable(n)
    u = np.array([epsilon/np.sqrt(2),epsilon/np.sqrt(2)])

    W_point = point.R.A
    W_point_prime = point_prime.R.A
    W_C = C.A

    b_point = point.R.b
    b_point_prime = point_prime.R.b
    b_C = C.b

    P = np.eye(n)
                                                        #@: Here, they use the intersection version 
    prob = cp.Problem(cp.Minimize(cp.sum(P)), #@: This minimizes the  norm of u
                      [z == z_point + u + c_point, 
                       z == z_point2 - c_point2,       #@: z is meant to be the intersection point
                       W_point @ z_point >= b_point,  #@: these two enforce the hyperrectangles
                       W_point_prime @ z_point2 >= b_point_prime,
                       W_C @ c_point >= b_C, #@: These two enforces c points to be from the cone
                       W_C @ c_point2 >= b_C])
                       #W_C @ u >= b_C])
    try:                   
        prob.solve(solver = "OSQP")
    except :#cp.error.SolverError:
        prob.solve(solver = "ECOS")

    # Print result.
    # print("\nThe optimal value is", prob.value)
    # print(u.value)

    condition = prob.status == 'optimal'  
    return condition


def ecovered_faster(point, point_prime, C, epsilon): 
    """

    :param point: DesignPoint x.
    :param point_prime: Design Point x'.
    :param C: Polyhedron C.
    :param epsilon:
    :return:
    """
    n = C.A.shape[1]


    z_point = cp.Variable(n)
    z_point2 = cp.Variable(n)
    u = np.array([epsilon/np.sqrt(2),epsilon/np.sqrt(2)])

    W_point = point.R.A
    W_point_prime = point_prime.R.A
    W_C = C.A

    b_point = point.R.b
    b_point_prime = point_prime.R.b
    b_C = C.b
    P = np.eye(n)
                                                        #@: Here, they use the intersection version 
    prob = cp.Problem(cp.Minimize(cp.sum(P)), #@: This minimizes the  norm of u
                      [ W_point @ z_point >= b_point,  #@: these two enforce the hyperrectangles
                       W_point_prime @ z_point2 >= b_point_prime,
                       W_C @ (z_point2-z_point-u)>= b_C])
    try:
        prob.solve(solver = "OSQP",max_iter=10000)#,verbose=True)
    except :#cp.error.SolverError:
        prob.solve(solver = "ECOS")

    # Print result.
    # print("\nThe optimal value is", prob.value)
    # print(u.value)

    if prob.status==None:
        return True

    condition = prob.status == 'optimal'  

    
    return condition




def find_farthest(point):

    # Brute force for now   @: Why not use point.R.diameter?
    vertices = point.R.get_vertices()

    no_vertices = len(vertices)
    largest = 0
    pair = None
    for i in range(no_vertices):
        for j in range(i, no_vertices):
            dist = np.linalg.norm(vertices[i] - vertices[j])

            if dist > largest:
                largest = dist
                pair = (i, j)

    return largest


def cone_order(x, y, cone):
    """
    Check if x <_C y
    :param point1: x
    :param point2: y
    :param cone: C
    :return:
    """
    W = cone.W
    z = y - x

    return np.all(W @ z > 0)

