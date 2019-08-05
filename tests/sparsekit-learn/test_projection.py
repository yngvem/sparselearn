"""
TODO: Implement KKT conditions test.
  -> That test will involve a significant increase in overall code complexity that
     might not be worth the extra test.
"""

from abc import ABC, abstractmethod
import itertools

import numpy as np
import pytest

from sparselearn import projections


def perturb_normally(x, rate):
    xnorm = np.linalg.norm(x)
    rate /= xnorm

    return x + rate*np.random.standard_normal(x.shape)


class BaseTestProjection(ABC):
    Projection = projections.BaseProjection
    projection_kwargs = [{}]

    @pytest.fixture
    def points(self):
        return [np.random.standard_normal((100,))*i for i in range(20)]

    def projections(self):
        """Iterate through all projections that should be tested.
        """
        for projection_kwargs in self.projection_kwargs:
            yield self.Projection(**projection_kwargs)
    
    @abstractmethod
    def is_feasible(self, x):
        pass

    def test_projects_to_interior(self, points):
        """Test that the projection in fact projects downto the feasible set.
        """
        for x, projection in itertools.product(points, self.projections()):
            y = projection(x)
            assert projection.is_feasible(y)

    def check_projects_to_minimum(self, x, projection):
        num_perturbations = 100
        y = projection(x)
        for _ in range(num_perturbations):
            y_pert = perturb_normally(y, 0.1)
            if projection.is_feasible(y_pert):
                assert np.linalg.norm(x - y) < np.linalg.norm(x - y_pert)

    def test_projects_to_minimum(self, points):
        """Test that the minimum is attained in the given point.
        """
        num_perturbations = 100

        for x, projection in itertools.product(points, self.projections()):
            self.check_projects_to_minimum(x, projection)

    def test_is_feasible_checks_feasibility(self, points):
        """Test that the is_feasible function of the projection works.
        """
        for x, kwargs in itertools.product(points, self.projection_kwargs):
            projection = self.Projection(**kwargs)
            y = projection(x)
            
            assert self.is_feasible(x, **kwargs) == projection.is_feasible(x)
            assert self.is_feasible(y, **kwargs) == projection.is_feasible(y)


class TestNonNegativeProjection(BaseTestProjection):
    Projection = projections.NonNegativityProjection

    def is_feasible(self, x):
        return np.all(x >= 0)


