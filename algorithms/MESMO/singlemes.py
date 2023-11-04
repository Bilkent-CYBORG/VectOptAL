import numpy as np
from scipy.stats import norm
from sklearn.kernel_approximation import RBFSampler


class MaxvalueEntropySearch(object):
    def __init__(self, GPmodel):
        self.GPmodel = GPmodel
        self.y_max = max(GPmodel.yValues)
        self.d = GPmodel.dim

    def Sampling_RFM(self):
        while True:
            try:
                self.rbf_features = RBFSampler(gamma=1/(2*self.GPmodel.kernel.length_scale**2), n_components=1000, random_state=1)
                X_train_features = self.rbf_features.fit_transform(np.asarray(self.GPmodel.xValues))

                A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(self.rbf_features.n_components)/self.GPmodel.beta)
                self.weights_mu = A_inv.dot(X_train_features.T).dot(self.GPmodel.yValues)
                weights_gamma = A_inv / self.GPmodel.beta
                self.L = np.linalg.cholesky(weights_gamma)
            except np.linalg.LinAlgError:
                pass
            else:
                break

    def weigh_sampling(self):
        random_normal_sample = np.random.normal(0, 1, np.size(self.weights_mu))
        self.sampled_weights = np.c_[self.weights_mu] + self.L.dot(np.c_[random_normal_sample])

    def f_regression(self,x):
        # Note that “fitting” the feature function does not actually depend on
        # the data given to the fit function. Only the dimensionality of the data is used.
        # https://scikit-learn.org/stable/modules/kernel_approximation.html#radial-basis-function-kernel
        X_features = self.rbf_features.fit_transform(x.reshape(1,len(x)))
        return -(X_features.dot(self.sampled_weights)) 

    def single_acq(self, x, maximum):
        mean, std = self.GPmodel.getPrediction(x)
        mean = mean[0]
        std = std[0]
        if maximum < max(self.GPmodel.yValues) + 5 / self.GPmodel.beta:
            maximum = max(self.GPmodel.yValues) + 5 / self.GPmodel.beta

        normalized_max = (maximum - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        if (cdf==0):
            cdf=1e-30
        return -(normalized_max * pdf) / (2*cdf) + np.log(cdf)
