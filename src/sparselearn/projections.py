from abc import ABC, abstractmethod
import numexpr as ne
import numpy as np


class BaseProjection(ABC):
    """Base class for orthogonal projections.
    """
    def __call__(self, x):
        return self.project(x)

    @abstractmethod
    def project(self, x):
        pass

    @abstractmethod
    def is_feasible(self, x):
        pass


class NonNegativityProjection(BaseProjection):
    """Project downto the nonnegative orthant.
    """
    def project(self, x):
        return ne.evaluate('x*(x >= 0)')

    def is_feasible(self, x):
        return np.all(x >= 0)


class SimplexProjection(BaseProjection):
    """Project downto the simplex.
    """
    def __init__(self, simplex_size):
        self.simplex_size = simplex_size
        self.non_negativity_projection = NonNegativityProjection()

    def project(self, x):
        y = self.non_negativity_projection.project(x)
        v = y
        rho = np.mean(v) - self.simplex_size

        while True:
            v_new = v[v > rho]
            rho = np.mean(v) - self.simplex_size

            if v == v_new:
                return np.maximum(y - rho, 0)


class BallProjection(BaseProjection):
    def __init__(self, max_norm):
        self.max_norm = max_norm


class L1Projection(BallProjection):
    """Project downto the L1 ball, i.2. :math:`||x||_1 < a`.
    """
    def __init__(self, max_norm):
        super().__init__(max_norm)
        self.simplex_projection = SimplexProjection(max_norm)

    def project(self, x):
        if self.is_feasible(x):
            return x
        return np.sign(x)*self.simplex_projection(np.abs(x))

    def is_feasible(self, x):
        return np.linalg.norm(x, 1) < self.max_norm


class L2Projection(BallProjection):
    """Project downto the L2 ball, i.e. :math:`||x||_2 < a`.
    """
    def project(self, x):
        xnorm = np.linalg.norm(x)
        if xnorm < self.max_norm:
            return x

        return (self.max_norm/xnorm)*x
    
    def is_feasible(self, x):
        return np.linalg.norm(x) < self.max_penalty


class LInfProjection(BallProjection):
    """Project downto the L-infinity ball, i.e. :math:`||x||_\infty < a`.
    """
    def project(self, x):
        return np.sign(x)*np.minimum(x, self.max_norm)

    def is_feasible(self, x):
        return np.linalg.norm(x, np.inf) < self.max_norm()

