import logging

import torch
import gpytorch

torch.set_default_dtype(torch.float64)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood,device):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.requires_grad_(False)
        self.device = device
        self.covar_module = kernel
        """ self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True) """

    def forward(self, x):
        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessModel:
    def __init__(self, m, d, kernel_list, noise_variance, x_sample = None, y_sample = None, verbose=False,device=None,train_during_alg = False, train_during_alg_iter=10):
        """
        Class for handling GP objects.
        :param d: Input dimension.
        :param m: Output dimension.
        :param kernel_list: Kernel list given in the beginning.
        :param verbose:
        :param train_during_alg: Whether to train GP during algorithm.
        :param train_during_alg_iter: Iteration of training of GP during algorithm.
        """
        self.device =device 
        self.train_during_alg = train_during_alg
        self.train_during_alg_iter = train_during_alg_iter
        #self.device = 'cpu'

        self.m = m
        self.d = d

        self.X = x_sample
        
        if x_sample is not None:
            if not isinstance(self.X, torch.Tensor):
                self.X = torch.tensor(self.X, dtype=torch.float64).to(self.device)
            self.Y = y_sample
            if not isinstance(self.Y, torch.Tensor):
                self.Y = torch.tensor(self.Y, dtype=torch.float64).to(self.device)

        # To initialize without data
        elif x_sample is None:
            self.X = torch.tensor([[-1e8] * self.d])
            self.Y = torch.tensor([[0.] * self.m])


        self.kernel_list = kernel_list
        self.verbose = verbose
        self.opt_models = []

        self.noise_variance = noise_variance
        
        # Set up likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-9))#.cuda() #.to(self.device)
        self.likelihood.noise = self.noise_variance
        self.likelihood.requires_grad_(False)
        
        self.model = self.single_output_gp_list(first=True)  # List of GP Models


    def single_output_gp_list(self, first=False):
        gp_list = []
        #print("Points sampled so far:\n", self.X, sep="")

        # Independent GP for each objective function
        for i in range(self.m):
            Y = self.Y[:, i]
            kernel = self.kernel_list[i]
            if self.device == "cuda":
                m = ExactGPModel(self.X, Y, kernel=kernel, likelihood=self.likelihood,device=self.device).cuda()
            else:
                m = ExactGPModel(self.X, Y, kernel=kernel, likelihood=self.likelihood,device=self.device)
            
            if first:
                # Optimize the model for meaningful predictions (maximize the log marginal likelihood)
                m.train()
                self.likelihood.train()

                training_iterations = 400
                optimizer = torch.optim.Adam(
                    m.parameters(), lr=1e-2
                )

                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, m)

                for e in range(training_iterations):
                    # Zero backprop gradients
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Get output from model
                    output = m(self.X,)

                    # Calculate loss and backprop derivatives
                    loss = -mll(output, Y)
                    loss.backward()
                    
                    print('Train iter %d/%d - Loss: %.3f' % (e + 1, training_iterations, loss.item()))
                    
                    optimizer.step()
            
                self.opt_models.append(m)

            elif self.train_during_alg: 
                m.train()
                self.likelihood.train()

                training_iterations = self.train_during_alg_iter
                optimizer = torch.optim.Adam(
                    m.parameters(), lr=1e-3 #2 de olabilir dene.
                )

                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, m)

                for e in range(training_iterations):
                    # Zero backprop gradients
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Get output from model
                    output = m(self.X,)

                    # Calculate loss and backprop derivatives
                    loss = -mll(output, Y)
                    loss.backward()
                    
                    #print('Train iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                    
                    optimizer.step()

                self.opt_models[i] = m

            else:
                m = ExactGPModel(
                    self.X, self.Y[:, i], kernel=self.opt_models[i].covar_module,
                    likelihood=self.likelihood,device = self.device
                )

            m.eval()
            self.likelihood.eval()

            
            gp_list.append(m)
            
        if first:

            random_index = torch.randint(0, self.X.shape[0], (1,))
            self.X =self.X[random_index.item()].reshape(-1, self.d)
            self.Y =self.Y[random_index.item()].reshape(-1, self.m)
            for ind,gp in enumerate(gp_list):
                gp.set_train_data(self.X, self.Y[:,ind], strict=False)



            """ self.X =self.X[:1]
            self.Y =self.Y[:1]  """
            
            #if self.verbose is True:
                #print("For objective function ", i)
                #print_summary(m)
                #print("Log likelihood ", tf.keras.backend.get_value(m.log_marginal_likelihood()))


        return gp_list




    def inference(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64).to(self.device)
        x = x.reshape(1, -1)

        mus = torch.empty((self.m, 1)) #np.empty((self.m, 1))
        var_sqrt = None  # np.empty((self.m, 1))
        cov = torch.zeros((self.m, self.m))

        with torch.no_grad():
            for i, gp in enumerate(self.model):
                m_post = gp(x)
                
                mus[i] = m_post.mean  #.mean ##.cpu().numpy()
                cov[i, i] = m_post.covariance_matrix ##.cpu().numpy()
        return mus, var_sqrt, cov



    def update(self, x, y):
        #print(x.shape)
        #print(self.X.shape)

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64).to(self.device).reshape(1, -1)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64).to(self.device).reshape(1, -1)

        self.X = torch.vstack((self.X, x))
        self.Y = torch.vstack((self.Y, y))

        self.model = self.single_output_gp_list(first=False)


class ExactGPModelDependent(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood,device,m):
        super(ExactGPModelDependent, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(),num_tasks=m)
        self.device = device
        self.covar_module = kernel
        
        self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True)

    def forward(self, x):
        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GaussianProcessModelDependent:
    def __init__(self, m, d, kernel, noise_variance, x_sample = None, y_sample = None, verbose=False,device=None,train_during_alg = False, train_during_alg_iter=10):
        """
        Class for handling GP objects.
        :param d: Input dimension.
        :param m: Output dimension.
        :param kernel: Kernel given in the beginning.
        :param verbose:
        :param train_during_alg: Whether to train GP during algorithm.
        :param train_during_alg_iter: Iteration of training of GP during algorithm.
        """
        self.device =device 
        self.train_during_alg = train_during_alg
        self.train_during_alg_iter = train_during_alg_iter
        #self.device = 'cpu'

        self.m = m
        self.d = d

        self.X = x_sample
        
        if x_sample is not None:
            if not isinstance(self.X, torch.Tensor):
                self.X = torch.tensor(self.X, dtype=torch.float64).to(self.device)
            self.Y = y_sample
            if not isinstance(self.Y, torch.Tensor):
                self.Y = torch.tensor(self.Y, dtype=torch.float64).to(self.device)

        # To initialize without data
        elif x_sample is None:
            self.X = torch.tensor([[-1e8] * self.d])
            self.Y = torch.tensor([[0.] * self.m])


        self.kernel = kernel
        self.verbose = verbose

        self.noise_variance = noise_variance
        
        # Set up likelihood and model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            self.m,
            rank=m,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-9),
            has_task_noise=False
        ).to(self.device)
        self.likelihood.noise = self.noise_variance
        self.likelihood.requires_grad_(False)
        self.model = None
        self.update(None, None)


    def inference(self, x):
        self.model.eval()
        self.likelihood.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64).to(self.device)
        x = x.reshape(1, -1)

        var_sqrt = None

        with torch.no_grad():
            m_post = self.model(x)
                
            mus = m_post.mean.squeeze().cpu().reshape(-1, 1)
            covariances = (m_post.covariance_matrix + m_post.covariance_matrix.transpose(0, 1)) / 2

            cov = covariances.cpu()

        return mus, var_sqrt, cov
    

    def inference_bulk(self, x):
        self.model.eval()
        self.likelihood.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64).to(self.device)

        var_sqrt = None

        with torch.no_grad():
            m_post = self.model(x)
                
            mus = m_post.mean.squeeze().cpu().reshape(-1, 1)
            covariances = (m_post.covariance_matrix + m_post.covariance_matrix.transpose(1, 2)) / 2

            cov = covariances.cpu()

        return mus, var_sqrt, cov


    def update(self, x, y):
        first = self.model is None

        

        if x is not None and y is not None:

            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float64).to(self.device).reshape(1, -1)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float64).to(self.device).reshape(1, -1)
            self.X = torch.vstack((self.X, x))
            self.Y = torch.vstack((self.Y, y))

        self.model = ExactGPModelDependent(self.X, self.Y, kernel=self.kernel, likelihood=self.likelihood, device=self.device,m=self.m)
        
        if first:
            # Optimize the model for meaningful predictions (maximize the log marginal likelihood)
            self.model.train()
            self.likelihood.train()

            training_iterations = 200
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=0.1
            )

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            for e in range(training_iterations):
                should_print = True if e == 0 or e == training_iterations-1 else False
                
                # Zero backprop gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Get output from model
                output = self.model(self.X)

                # Calculate loss and backprop derivatives
                loss = -mll(output, self.Y)
                loss.backward()
                
                log_text = 'Train iter %d/%d - Loss: %.3f' % (e + 1, training_iterations, loss.item())
                if should_print:
                    logging.info(log_text)
                    should_print = False
                
                optimizer.step()
            
            self.model.eval()
            self.likelihood.eval()
        else:
            self.model.set_train_data(self.X, self.Y, strict=False)

        if first:
            random_index = torch.randint(0, self.X.shape[0], (1,))
            self.X = self.X[random_index.item()].reshape(-1, self.d)
            self.Y = self.Y[random_index.item()].reshape(-1, self.m)
            self.model.set_train_data(self.X, self.Y, strict=False)

       