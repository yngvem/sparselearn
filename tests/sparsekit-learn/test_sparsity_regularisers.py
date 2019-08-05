from abc import ABC, abstractmethod
import numpy as np
import pytest

from sparselearn import regularisers


class BaseTestRegulariser:
    Regulariser = None
    regulariser_kwargs = {}

    @abstractmethod
    def random_x(self):
        pass

    @abstractmethod
    def regulariser_loss(self, x, regulariser_kwargs):
        pass

    @pytest.fixture
    def regulariser(self):
        return self.Regulariser(**self.regulariser_kwargs)

    def test_neighbouring_points_have_larger_proximal_loss(self, regulariser):
        """
        Test that the prox operator gives lower prox-loss than 100 random points.
        """
        x = self.random_x()
        y = regulariser.prox(x)

        for i in range(100):
            y_perturbed = y + 20*np.random.standard_normal(y.shape)/(i+1)**2
            assert regulariser.prox_loss(y, x) < regulariser.prox_loss(y_perturbed, x)

    def test_prox_subgradient_relation(self, regulariser):
        """
        Test that the prox operator gives a minimum according to subgradient.

        This test might fail if the subgradient method of the regulariser
        doesn't return 0 in a minimum.
        """
        for i in range(100):
            x = self.random_x()
            y = regulariser.prox(x)
            assert np.allclose(regulariser.subgradient(y) + y, x)

    def test_subgradient_underestimates_regulariser(self, regulariser):
        """
        Test that the subgradient-based linearisation underestimates the regulariser
        """
        for i in range(100):
            x = self.random_x()
            intercept = regulariser(x)
            slope = regulariser.subgradient(x)

            for j in range(100):
                y = self.random_x()
                
                assert regulariser(y) > intercept + slope.T@(y - x)
