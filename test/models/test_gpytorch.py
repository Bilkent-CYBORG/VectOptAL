import unittest
import torch
import numpy as np
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood

from vectoptal.models.gpytorch import (
    MultitaskExactGPModel,
    BatchIndependentExactGPModel,
    GPyTorchMultioutputExactModel,
    CorrelatedExactGPyTorchModel,
    IndependentExactGPyTorchModel,
    SingleTaskGP,
    GPyTorchModelListExactModel,
    get_gpytorch_modellist_w_known_hyperparams,
)

from vectoptal.maximization_problem import Problem


class TestMultitaskExactGPModel(unittest.TestCase):
    """Tests for the MultitaskExactGPModel class."""

    def setUp(self):
        """Set up sample data and model for tests."""
        self.X = torch.randn(10, 2)
        self.Y = torch.randn(10, 3)
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.Y.shape[-1])
        self.model = MultitaskExactGPModel(self.X, self.Y, self.likelihood, RBFKernel)

    def test_forward(self):
        """Test forward pass of MultitaskExactGPModel."""
        output = self.model(self.X)
        self.assertIsInstance(output, MultitaskMultivariateNormal, "Output type mismatch.")


class TestBatchIndependentExactGPModel(unittest.TestCase):
    """Tests for the BatchIndependentExactGPModel class."""

    def setUp(self):
        """Set up sample data and model for tests."""
        self.X = torch.randn(10, 2)
        self.Y = torch.randn(10, 3)
        self.likelihood = GaussianLikelihood()
        self.model = BatchIndependentExactGPModel(self.X, self.Y, self.likelihood, RBFKernel)

    def test_forward(self):
        """Test forward pass of BatchIndependentExactGPModel."""
        output = self.model(self.X)
        self.assertIsInstance(output, MultitaskMultivariateNormal, "Output type mismatch.")


class TestableGPyTorchMultioutputExactModel(GPyTorchMultioutputExactModel):
    """
    Subclass of GPyTorchMultioutputExactModel to implement the abstract predict method for testing.
    """

    def predict(self, X):
        """Mock predict implementation for testing."""
        return np.zeros((X.shape[0], self.output_dim)), np.zeros(
            (X.shape[0], self.output_dim, self.output_dim)
        )


class TestGPyTorchMultioutputExactModel(unittest.TestCase):
    """Tests for the GPyTorchMultioutputExactModel class."""

    def setUp(self):
        """Set up model instance with test data."""
        # Use the testable subclass to avoid abstract class issues
        self.model = TestableGPyTorchMultioutputExactModel(
            input_dim=2, output_dim=3, noise_var=0.1, model_kind=MultitaskExactGPModel
        )

        # Sample test data
        self.X = torch.randn(5, 2)
        self.Y = torch.randn(5, 3)

        # Add sample and update model
        self.model.add_sample(self.X, self.Y)
        self.model.update()

    def test_add_sample(self):
        """Test add_sample method."""
        self.assertEqual(self.model.train_inputs.shape[0], 5)
        self.assertEqual(self.model.train_targets.shape[0], 5)

    def test_clear_data(self):
        """Test clear_data method."""
        self.model.clear_data()
        self.assertEqual(self.model.train_inputs.shape[0], 0)
        self.assertEqual(self.model.train_targets.shape[0], 0)

    def test_evaluate_kernel(self):
        """Test evaluate_kernel method."""
        kernel_matrix = self.model.evaluate_kernel(self.X)
        self.assertIsInstance(kernel_matrix, np.ndarray, "Kernel matrix type mismatch.")


class TestCorrelatedExactGPyTorchModel(unittest.TestCase):
    """Tests for the CorrelatedExactGPyTorchModel class."""

    def setUp(self):
        """Set up correlated multitask model."""
        self.model = CorrelatedExactGPyTorchModel(input_dim=2, output_dim=3, noise_var=0.1)
        self.X = torch.randn(5, 2)
        self.Y = torch.randn(5, 3)
        self.model.add_sample(self.X, self.Y)
        self.model.update()

    def test_predict(self):
        """Test predict method."""
        means, variances = self.model.predict(self.X)
        self.assertIsInstance(means, np.ndarray, "Means type mismatch.")
        self.assertIsInstance(variances, np.ndarray, "Variances type mismatch.")


class TestIndependentExactGPyTorchModel(unittest.TestCase):
    """Tests for the IndependentExactGPyTorchModel class."""

    def setUp(self):
        """Set up independent multitask model."""
        self.model = IndependentExactGPyTorchModel(input_dim=2, output_dim=3, noise_var=0.1)
        self.X = torch.randn(5, 2)
        self.Y = torch.randn(5, 3)
        self.model.add_sample(self.X, self.Y)
        self.model.update()

    def test_predict(self):
        """Test predict method."""
        means, variances = self.model.predict(self.X)
        self.assertIsInstance(means, np.ndarray, "Means type mismatch.")
        self.assertIsInstance(variances, np.ndarray, "Variances type mismatch.")


class TestSingleTaskGP(unittest.TestCase):
    """Tests for the SingleTaskGP class."""

    def setUp(self):
        """Set up single-task GP model."""
        self.X = torch.randn(10, 2)
        self.Y = torch.randn(10)
        self.likelihood = GaussianLikelihood()
        self.model = SingleTaskGP(self.X, self.Y, self.likelihood, RBFKernel)

    def test_forward(self):
        """Test forward pass of SingleTaskGP."""
        output = self.model(self.X)
        self.assertIsInstance(output, MultivariateNormal, "Output type mismatch.")


class TestGPyTorchModelListExactModel(unittest.TestCase):
    """Tests for the GPyTorchModelListExactModel class."""

    def setUp(self):
        """Set up multi-output GP model list."""
        self.model = GPyTorchModelListExactModel(input_dim=2, output_dim=3, noise_var=0.1)
        self.X = torch.randn(5, 2)
        self.Y = torch.randn(5, 3)
        for i in range(self.Y.shape[1]):
            self.model.add_sample(self.X, self.Y[:, i], dim_index=i)
        self.model.update()

    def test_predict(self):
        """Test predict method."""
        means, variances = self.model.predict(self.X)
        self.assertIsInstance(means, np.ndarray, "Means type mismatch.")
        self.assertIsInstance(variances, np.ndarray, "Variances type mismatch.")

    def test_sample_from_posterior(self):
        """Test sample_from_posterior method."""
        samples = self.model.sample_from_posterior(self.X, sample_count=3)
        self.assertEqual(samples.shape[0], 3, "Sample count mismatch.")

    def test_sample_from_single_posterior(self):
        """Test sample_from_single_posterior method."""
        samples = self.model.sample_from_single_posterior(self.X, dim_index=1, sample_count=2)
        self.assertEqual(samples.shape[0], 2, "Sample count mismatch for single posterior.")


class TestGetGPyTorchModelListWithKnownHyperparams(unittest.TestCase):
    """Test the `get_gpytorch_modellist_w_known_hyperparams` function."""

    def setUp(self):
        """Set up problem instance with test data."""
        self.input_dim = 2
        self.output_dim = 2
        self.noise_var = 0.1
        self.initial_sample_cnt = 5

        # Mock problem to provide in_dim and evaluate function
        self.problem = unittest.mock.Mock(spec=Problem)
        self.problem.in_dim = self.input_dim

    def test_get_model_without_initial_data(self):
        """Test model creation when X and Y are not provided."""
        self.problem.evaluate.return_value = np.random.randn(1024, self.output_dim)

        model = get_gpytorch_modellist_w_known_hyperparams(
            problem=self.problem,
            noise_var=self.noise_var,
            initial_sample_cnt=self.initial_sample_cnt,
        )

        # Check if the returned object is an instance of GPyTorchModelListExactModel
        self.assertIsInstance(model, GPyTorchModelListExactModel)

        # Ensure that the problem's evaluate method was called (to generate initial Y)
        self.assertGreaterEqual(self.problem.evaluate.call_count, 1)
        total_samples = 0
        for dim in range(self.output_dim):
            total_samples += model.train_inputs[dim].shape[0]
        self.assertEqual(total_samples, self.initial_sample_cnt)

    def test_get_model_with_initial_data(self):
        """Test model creation when X and Y are provided."""
        self.problem.evaluate.return_value = np.random.randn(
            self.initial_sample_cnt, self.output_dim
        )

        X = np.random.randn(10, self.input_dim)
        Y = np.random.randn(10, self.output_dim)

        model = get_gpytorch_modellist_w_known_hyperparams(
            problem=self.problem,
            noise_var=self.noise_var,
            initial_sample_cnt=self.initial_sample_cnt,
            X=X,
            Y=Y,
        )

        self.assertEqual(self.problem.evaluate.call_count, 1)
        # Check that the correct number of initial samples were added
        self.assertEqual(len(model.train_inputs), self.output_dim)

        total_samples = 0
        for dim in range(self.output_dim):
            total_samples += model.train_inputs[dim].shape[0]
        self.assertEqual(total_samples, self.initial_sample_cnt)
