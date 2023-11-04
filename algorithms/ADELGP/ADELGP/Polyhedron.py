import numpy as np
import cvxpy as cp
#import cdd


class Polyhedron:
    # TODO: Change to hyperrectangle class

    def __init__(self, A=None, b=None):
        """
        Polyhedron in the format P = {x: Ax > b}
        :param A:
        :param b:
        """
        self.A = A
        self.b = b

        if A is None:
            self.A = None
            self.b = None
            self.lower = None
            self.upper = None

        self.vertices = None

        if self.b is not None:
            self.lower = self.b[0:2]
            self.upper = -self.b[2:4]


    def update_vrep(self):

        ineq = np.concatenate([self.b[:, None], self.A], axis=1)

        mat = cdd.Matrix(ineq.tolist(), number_type='fraction')
        poly = cdd.Polyhedron(mat)

        ext = poly.get_generators()
        ext = ext.__getitem__(slice(0, ext.row_size + 1))
        b = [list(x) for x in ext]
        npa = np.asarray(b, dtype=np.float64)
        self.vertices = npa[:, 1:]
        return npa[:, 1:]


    def diameter(self):
        return np.linalg.norm(self.upper - self.lower)


    def remove_redundant(self):
        i = 0
        while i < len(self.A) and 1 < len(self.A):
            M = np.delete(-self.A, i, axis=0)
            n = np.delete(-self.b, i, axis=0)

            s = -self.A[i]
            t = -self.b[i]

            x = cp.Variable(self.A.shape[1])
            prob = cp.Problem(cp.Maximize(s.T @ x),
                              [M @ x <= n, s.T @ x <= t + 1])
            prob.solve()

            i += 1

            if prob.value <= t or np.isclose(prob.value, t):
                self.A = M
                self.b = n
                i -= 1


    def intersect(self, polyhedron, set_itself=False):
        if self.A is None:
            self.A = polyhedron.A
            self.b = polyhedron.b
        else:
            A = np.concatenate([self.A, polyhedron.A])
            b = np.concatenate([self.b, polyhedron.b])

            if set_itself:
                self.A = A
                self.b = b

            return A, b


    def __str__(self):
        if self.A is None:
            return "empty"
        else:
            return "A \n" + np.array2string(self.A) + "\nb \n" + np.array2string(self.b)


    def __eq__(self, other):
        return self.A == other.A and self.b == other.b


    def get_vertices(self):
        return self.vertices
