from unittest import TestCase, mock

import torch
import numpy as np

from vectoptal.utils import (
    set_seed,
    get_2d_w,
    get_alpha,
    get_alpha_vec,
    get_closest_indices_from_points,
    get_noisy_evaluations_chol,
    generate_sobol_samples,
    get_smallmij,
    get_delta,
)


class TestSetSeed(TestCase):
    """Test seed setting."""

    def test_set_seed(self):
        """Test the set_seed function."""
        seeds = [0, 42, 123]
        for seed in seeds:
            with self.subTest(seed=seed):
                set_seed(seed)

                self.assertEqual(np.random.get_state()[1][0], seed)
                self.assertListEqual(
                    torch.random.get_rng_state().tolist(),
                    torch.manual_seed(seed).get_state().tolist(),
                )


class TestGet2DW(TestCase):
    """Test 2D cone matrix generation."""

    def test_get_2d_w(self):
        """Test the get_2d_w function."""
        cone_angles = [45, 60, 90, 120, 135]
        for cone_angle in cone_angles:
            with self.subTest(cone_angle=cone_angle):
                W = get_2d_w(cone_angle)

                for i in range(2):
                    self.assertAlmostEqual(np.linalg.norm(W[i]), 1.0)

                self.assertAlmostEqual(-np.dot(W[0], W[1]), np.cos(np.deg2rad(cone_angle)))


class TestGetAlpha(TestCase):
    """Test alpha computation."""

    def test_get_alpha(self):
        """Test the get_alpha function."""
        angles = [45, 60, 90, 120, 135]
        for angle in angles:
            cos_ang = np.cos(np.deg2rad(max(90 - angle, 0)))
            W = get_2d_w(angle)
            for rind in range(W.shape[0]):
                with self.subTest(rind=rind):
                    alpha = get_alpha(rind, W)
                    self.assertAlmostEqual(alpha, cos_ang)

    def test_get_alpha_vec(self):
        """Test the get_alpha_vec function."""
        angle = 45
        cos_ang = np.cos(np.deg2rad(max(90 - angle, 0)))
        W = get_2d_w(angle)
        alphas = get_alpha_vec(W)

        np.testing.assert_allclose(alphas, [[cos_ang]] * W.shape[0])


class TestGetClosestIndicesFromPoints(TestCase):
    """Test closest indices computation."""

    def test_get_closest_indices_from_points(self):
        """Test the get_closest_indices_from_points function."""
        points = torch.tensor([[0, 0], [1, 1]])
        queries = torch.tensor([[0.1, 0.1], [0.5, 0.9], [1.2, 1.2]])

        result_sq = get_closest_indices_from_points(
            queries, points, return_distances=False, squared=True
        )
        result = get_closest_indices_from_points(
            queries, points, return_distances=False, squared=False
        )
        self.assertListEqual(result_sq.tolist(), result.tolist())
        self.assertListEqual(result.tolist(), [0, 1, 1])

        result_sq, dists_sq = get_closest_indices_from_points(
            queries, points, return_distances=True, squared=True
        )
        result, dists = get_closest_indices_from_points(
            queries, points, return_distances=True, squared=False
        )

        self.assertListEqual(result_sq.tolist(), result.tolist())
        self.assertListEqual(result.tolist(), [0, 1, 1])
        np.testing.assert_allclose(dists_sq, dists**2)
        np.testing.assert_allclose(dists_sq, [0.02, 0.26, 0.08])


class TestGetNoisyEvaluationsChol(TestCase):
    """Test noisy evaluations generation."""

    @mock.patch("vectoptal.utils.utils.np.random.normal")
    def test_get_noisy_evaluations_chol(self, mock_normal):
        """Test the get_noisy_evaluations_chol function."""
        n = 10
        x = np.linspace(0, 1, n).reshape(-1, 1)
        y = np.sin(2 * np.pi * x)

        mock_normal.return_value = x
        np.testing.assert_allclose(get_noisy_evaluations_chol(y, np.zeros((1, 1))), y)

        sigma = np.ones((1, 1)) * 0.1
        y_noisy = get_noisy_evaluations_chol(y, sigma)
        self.assertEqual(y_noisy.shape, y.shape)
        np.testing.assert_allclose(y_noisy, y + sigma * x)


class TestGenerateSobolSamples(TestCase):
    """Test Sobol samples generation."""

    def test_generate_sobol_samples(self):
        """Test the generate_sobol_samples function."""
        dim = 2
        n = 16
        samples = generate_sobol_samples(dim, n)

        self.assertEqual(samples.shape, (n, dim))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < 1))
        self.assertEqual(len(np.unique(samples, axis=0)), n)


class TestGetSmallmij(TestCase):
    """Test m(i, j) computation."""

    def test_get_smallmij(self):
        """Test the get_smallmij function."""
        vi = np.array([1, 0])
        vj = np.array([1.1, 0.2])

        angles = [45, 60, 90, 120, 135]
        for angle in angles:
            with self.subTest(angle=angle):
                W = get_2d_w(angle)
                alpha_vec = get_alpha_vec(W)

                W_normalized = (W.T / alpha_vec.flatten()).T
                diff = vj - vi

                m = get_smallmij(vi, vj, W, alpha_vec)
                np.testing.assert_allclose(
                    m, min(np.clip(W_normalized @ diff, a_min=0, a_max=None))
                )


class TestGetDelta(TestCase):
    """Test delta gap computation."""

    def test_get_delta(self):
        """Test the get_delta function."""
        means = np.array([[1, 0], [1.1, 0.2], [0.5, 0.8]])

        angles = [45, 60, 90, 120, 135]
        for angle in angles:
            with self.subTest(angle=angle):
                W = get_2d_w(angle)
                alpha_vec = get_alpha_vec(W)

                delta_true = get_delta(means, W, alpha_vec)
                delta_expected = np.zeros_like(delta_true)
                for i in range(means.shape[0]):
                    vi = means[i]
                    for j in range(means.shape[0]):
                        vj = means[j]
                        mij = get_smallmij(vi, vj, W, alpha_vec)
                        delta_expected[i] = max(delta_expected[i], mij)
                        self.assertLessEqual(mij, delta_true[i])
                np.testing.assert_allclose(delta_true, delta_expected)
