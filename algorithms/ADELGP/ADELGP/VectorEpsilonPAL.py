from algorithms.ADELGP.ADELGP.phases import *
from algorithms.ADELGP.ADELGP.Polyhedron import Polyhedron
from algorithms.ADELGP.ADELGP.Hyperrectangle import Hyperrectangle
from algorithms.ADELGP.ADELGP.DesignPoint import DesignPoint
from algorithms.ADELGP.ADELGP.utils import *
from algorithms.ADELGP.ADELGP.utils_plot import *


class VectorEpsilonPAL:

    def __init__(self, problem_model, cone, epsilon, delta, gp, obj_dim, maxiter=None,batched = False):
        """
        VectorEpsilonPAL object.
        :param problem_model: OptimizationProblem object.
        :param cone: Polyhedron object.
        :param epsilon: epsilon parameter.
        :param delta: delta parameter.
        :param gp: GaussianProcessModel object.
        :param obj_dim: Objective space dimension.
        :param maxiter: Maximum iteration.
        """

        self.problem_model = problem_model
        self.gp = gp
        self.cone = cone
        self.epsilon = epsilon
        self.delta = delta
        self.maxiter=maxiter
        self.batched = batched
        # Rounds
        self.t = 0  # Total number of iterations


        self.sample_count = 0

        # Sets
        self.P = []  # Decided design points
        self.S = [DesignPoint(row, Hyperrectangle(upper=[np.inf]*self.gp.m, lower=[-np.inf]*self.gp.m),design_index=i) for i,row in enumerate(problem_model.x)]  # Undecided design points
        self.beta = np.ones(obj_dim, )



    def algorithm(self):
        """
        vector-epsilon-PAL algorithm.
        :return: List of DesignPoint objects.
        """
        # The region is a hyper-rectangle, set the cone as R+
        """ A_matrix = np.eye(self.gp.m)
        b_vector = np.array([0] * self.gp.m)
        cone = Polyhedron(A=A_matrix, b=b_vector) """

        while len(self.S) != 0:  # While S_t is not empty
            print(f"Round {self.t}")
            # Active nodes, union of sets s_t and p_t at the beginning of round t
            A = self.P + self.S

            "Modeling"
            # Set beta for this round

            self.beta = self.find_beta()
            modeling(A, self.gp, self.beta, self.cone, self.t)  # TODO: Change this to hyperrectangle class


            "Discarding"
            discard(self.S, self.P, self.cone, self.epsilon)


            "epsilon-Covering"
            # The union of sets S and P at the beginning of epsilon-Covering
            W = self.S + self.P
            epsiloncovering(self.S, self.P, self.cone, self.epsilon)


            "Evaluating"
            if self.S:  # If S_t is not empty
                x = evaluate(W,self.gp,self.t,self.beta,self.cone,self.batched)
                self.sample_count += len(x)
                for design in x:
                    y = self.problem_model.observe(design.x)

                    self.gp.update(design.x, y)
            
            if self.t == self.maxiter:
                return self.P

            self.t += 1

            print(f"There are {len(self.S)} designs left in set S.")
        return self.P


    def find_beta(self):
        beta = (2/20) * np.log(2*self.gp.m * self.problem_model.cardinality * (np.pi ** 2) * ((self.t+2) ** 2) / (3 * self.delta)) #This is according to the proofs.
        #beta = (8/80) * np.log(self.gp.m * self.problem_model.cardinality * np.pi ** 2 * t ** 2 / (6 * self.delta))

        return beta * np.ones(self.gp.m, )