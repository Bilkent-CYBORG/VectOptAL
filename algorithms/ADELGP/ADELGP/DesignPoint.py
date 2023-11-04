import numpy as np

from algorithms.ADELGP.ADELGP.Hyperrectangle import Hyperrectangle


class DesignPoint:

    def __init__(self, x: np.ndarray, R: Hyperrectangle, design_index: int):
        self.x = x
        self.R = R  # The confidence region (Hyperrectangle)
        self.design_index = design_index

    def __eq__(self, other):
        return (self.x == other.x).all()

    def __str__(self):
        name = "\nDesign Point: x " + str(self.x) +\
               "\nHyperrectangle" + str(self.R)
        return name

    def update_cumulative_conf_rect(self, mu, cov, beta, t):
        # High probability lower and upper bound, B
        L = mu.reshape(-1)-np.sqrt(np.diag(cov))*np.sqrt(beta)#mu - np.sqrt(beta) * sigma
        U = mu.reshape(-1)+np.sqrt(np.diag(cov))*np.sqrt(beta)#mu + np.sqrt(beta) * sigma

        #L = mu.reshape(-1)-np.sqrt(np.diag(cov))*np.log(beta)#mu - np.sqrt(beta) * sigma
        #U = mu.reshape(-1)+np.sqrt(np.diag(cov))*np.log(beta)#mu + np.sqrt(beta) * sigma
        
        # Confidence hyperrectangle, Q
        Q = Hyperrectangle(L.reshape(-1).tolist(), U.reshape(-1).tolist())
        # Cumulative confidence hyperrectangle, R
        self.R = self.R.intersect(Q,t)
        self.mu = mu
        #self.cov = cov
        

