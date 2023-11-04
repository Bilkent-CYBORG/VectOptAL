import torch


class OptimizationProblem:

    def __init__(self, x, y, obs_noise_std=0):
        self.x = x
        self.y = y
        self.cardinality = len(self.x)
        self.obs_noise_std = obs_noise_std


    def observe(self, point,remove_noise = False):
        if point.dim() == 1:
            index = torch.where(torch.all(self.x==point,axis=1))[0].item()
            return self.y[index] + torch.distributions.MultivariateNormal(torch.zeros(self.y[index].size()[0]), torch.eye(self.y[index].size()[0])).rsample() * self.obs_noise_std*(not remove_noise)
        elif point.dim() >1:
            indexes = [torch.where(torch.all(self.x== row,axis=1))[0].item() for row in point]
            return self.y[indexes] + torch.distributions.MultivariateNormal(torch.zeros(self.y[indexes].size()[0]), torch.eye(self.y[indexes].size()[0])).rsample() * self.obs_noise_std*(not remove_noise)

