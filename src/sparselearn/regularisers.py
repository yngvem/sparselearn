from abc import ABC, abstractmethod
import numbers
from typing import NamedTuple

import numpy as np


class BaseSparseRegulariser(ABC):
    """Base class for sparsity-based regularisers.
    """
    def __init__(self, reg):
        """
        Initialise regulariser.
        """
        self.reg = reg

    @abstractmethod
    def __call__(self, x):
        """
        Evaluate penalty.
        """
        pass

    @abstractmethod
    def subgradient(self, x):
        """
        A weak subgradient calculus of the regulariser.
        """
        pass

    @abstractmethod
    def prox(self, x):
        """
        The proximal operator for the regulariser.
        """
        pass

    def prox_loss(self, x, y):
        """
        The proximal loss :math:`R(x) + 0.5||x - y||^2`.
        """
        return self.__call__(x) + 0.5*np.linalg.norm(x - y)


# Lasso
class L1Regularisation(BaseSparseRegulariser):
    """(Weighted) L1 regularisation.
    """
    def __init__(self, reg):
        """
        Initialise regulariser.

        The regularisation coefficient can either be a single number or an array 
        containing coefficient weights. If it is a weight-array, then its shape
        must be equal to that of the coefficient vector.

        Arguments
        ---------
        reg : float or np.ndarray
            The regularisation coefficient(s).
        """
        self.reg = reg

    def __call__(self, x):
        """
        Evaluate the L1 penalty.

        Arguments
        ---------
        x : np.ndarray
        """
        return np.linalg.norm((weights*x).ravel(), 1)

    def subgradient(self, x):
        """
        Evaluate the subgradient.

        Arguments
        ---------
        x : np.ndarray
        """
        return np.sign(x)

    def prox(self, x):
        """
        Evaluate the proximal operator.

        Arguments
        ---------
        x : np.ndarray
        """
        return self.sign(x)*np.maximum(np.abs(x) - self.reg, 0)


# GroupLasso
class GroupMetaInfo(NamedTuple):
    x_g : np.ndarray
    reg : float
    idxes : np.ndarray


class GroupLassoRegularisation(BaseSparseRegulariser):
    """
    Group lasso regularisation with non-overlapping groups.
    """
    def __init__(self, reg, groups=None):
        """
        Initialise the non-overlappin groups group lasso regulariser.

        Initialise the group lasso regulariser with given regularisation
        coefficient and groups. The regularisation coefficient can either be
        an indexable object so that ``reg[group]`` is valid for all elements
        ``group in np.unique(groups)``, or a single number. If ``reg`` is a
        single number, then the regularisation coefficient for group :math:`g`,
        :math:`r_g` is set to :math:`r\sqrt{n_g}`, where :math:`r` is the overall
        regularisation coefficient and :math:`n_g` is the number of elements
        in group :math:`g`.

        If groups are set to None, then it each feature is stored in a different group.
        That is, each row of the coefficient matrix is its own group.

        Arguments
        ---------
        reg : float or Iterable
        groups : Iterable or None
        """
        self.groups = np.asarray(groups)
        self.reg = reg

    def iter_groups(self, x):
        """
        Iterate through the groups and return group metainfo.

        Arguments
        ---------
        x : np.ndarray
            Coefficient array
        """
        if self.groups is None:
            for i, x_g in enumerate(x):
                if isinstance(self.reg, numbers.Number):
                    reg = self.reg
                else:
                    reg = self.reg[i]

                yield GroupMetaInfo(x_g, reg, i)

        for group in np.unique(self.groups):
            idxes = self.groups == group
            x_g = x[idxes]
            if isinstance(self.reg, numbers.Number):
                reg = sqrt(idxes.sum())*self.reg
            else:
                reg = self.reg[group]
            yield GroupMetaInfo(x_g, reg, idxes)

    def __call__(self, x):
        """
        Evaluate the group lasso penalty.

        Arguments
        ---------
        x : np.ndarray
        """
        loss = 0
        for x_g, reg, _ in self.iter_groups():
            loss += reg*np.linalg.norm(x_g)

    def subgradient(self, x):
        """
        Evaluate the subgradient.

        Arguments
        ---------
        x : np.ndarray
        """
        subgradient = np.empty_like(x)
        for x_g, reg, idxes in self.iter_groups():
            xnorm = np.linalg.norm(x_g)
            if xnorm < 1e-16:
                subgradient[idxes] = 0
                continue

            subgradient[idxes] = (reg/xnorm)*x_g

    def prox(self, x):
        """
        Evaluate the proximal operator.

        Arguments
        ---------
        x : np.ndarray
        """
        y = np.empty_like(subgradient)
        for x_g, reg, idxes in self.iter_groups():
            xnorm = np.linalg.norm(x_g)
            if xnorm < reg:
                y[idxes] = 0
                continue
            
            y[idxes] = x_g - (reg/xnorm)*x_g
