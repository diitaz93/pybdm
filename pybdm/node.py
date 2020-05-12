"""_Algorithms based on ``BDM`` objects."""
from itertools import product
from random import choice
from .exceptions import NDimMismatch
import numpy as np


class NodePerturbationExperiment:
    """Node Perturbation experiment class.

    Node perturbation experiment studies change of BDM / entropy of a 
    network (given their adjacency matrix) when existing nodes are removed.
    This is the main tool for detecting nodes of a network having some 
    causal significance as opposed to noise parts.

    Nodes which when removed yield negative contribution to the overall
    complexity after change are likely to be important for the system,
    since their removal make it more noisy. On the other hand nodes that yield
    positive contribution to the overall complexity after removal are likely
    to be noise since they elongate the system's description length.

    Attributes
    ----------
    bdm : BDM
        BDM object. It has to be configured properly to handle
        the dataset that is to be studied.
    X : array_like (optional)
        Dataset for perturbation analysis. May be set later.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.
    binary_network: boolean
            If ``False``, the dataset (X) is the adjacency matrix of a one-type node
            network where all nodes can connect to each other, otherwise the matrix 
            represents a binary (two-type node) network where only pairs of nodes 
            of different types can connect, where nodes from one type are in the
            rows and from the other type in the columns.

    See also
    --------
    pybdm.bdm.BDM : BDM computations
#
    Examples
    --------
    >>> import numpy as np
    >>> from pybdm import BDM, NodePerturbationExperiment
    >>> X = np.random.randint(0, 2, (100, 100))
    >>> bdm = BDM(ndim=2)
    >>> npe = NodePerturbationExperiment(bdm, metric='bdm')
    >>> npe.set_data(X)

    >>> idx = np.argwhere(X) # Perturb only ones (1 --> 0)
    >>> delta_bdm = pe.run(idx)
    >>> len(delta_bdm) == idx.shape[0]
    True

    More examples can be found in :doc:`usage`.
    """
    def __init__(self, bdm, X=None, metric='bdm', binary_network=False):
        """Initialization method."""
        self.bdm = bdm
        self.metric = metric
        self.binary_network = binary_network
        #self._counter = None
        self._value = None
        #self._ncounts = None
        if self.metric == 'bdm':
            self._method = self._update_bdm
        elif self.metric == 'ent':
            self._method = self._update_ent
        else:
            raise AttributeError("Incorrect metric, not one of: 'bdm', 'ent'")
        if X is None:
            self.X = X
        else:
            self.set_data(X)
        if not self.ndim==2:
            raise ValueError("The number of dimensions of input dataset has to be 2")

    def __repr__(self):
        cn = self.__class__.__name__
        bdm = str(self.bdm)[1:-1]
        return "<{}(metric={}) with {}>".format(cn, self.metric, bdm)

    @property
    def size(self):
        """Data size getter."""
        return self.X.size

    @property
    def shape(self):
        """Data shape getter."""
        return self.X.shape

    @property
    def ndim(self):
        """Data number of axes getter."""
        return self.X.ndim

    def set_data(self, X):
        """Set dataset for the perturbation experiment.

        Parameters
        ----------
        X : array_like
            Dataset to perturb.
        """
        if not np.isin(np.unique(X), range(self.bdm.nsymbols)).all():
            raise ValueError("'X' is malformed (too many or ill-mapped symbols)")
        self.X = X
        #self._counter = self.bdm.decompose_and_count(X)
        if self.metric == 'bdm':
            self._value = self.bdm.bdm(self.X)
        elif self.metric == 'ent':
            self._value = self.bdm.ent(self._counter)
            #self._ncounts = sum(self._counter.values())

    def _update_bdm(self, idx, axis, keep_changes, binary_network):
        old_bdm = self._value
        if binary_network:
            newX = np.delete(X,idx,axis=0)
            newX = np.delete(newX,idx,axis=1)
        else:
            newX = np.delete(X,idx,axis=axis)
        new_bdm = self.bdm.bdm(newX)
        if keep_changes:
            self.X = newX
            self._value = new_bdm
        return new_dbm - old_bdm

    def _update_ent(self, idx, axis, keep_changes, binary_network):
        old_ent = self._value
        if binary_network:
            newX = np.delete(X,idx,axis=0)
            newX = np.delete(newX,idx,axis=1)
        else:
            newX = np.delete(X,idx,axis=axis)
        new_ent = self.bdm.ent(newX)
        if keep_changes:
            self.X = newX
            self._value = new_ent
        return new_ent - old_ent

    def perturb(self, idx, axis=0, keep_changes=False, binary_network=False):
        """Perturb element of the dataset.

        Parameters
        ----------
        idx : tuple
            Index tuple of an element.
        axis : int
            If binary_network is ``True``, is the axis that is used to remove node,
            otherwise is ignored.
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.
        binary_network: boolean
            If ``False``, the dataset (X) is the adjacency matrix of a one-type node
            network where all nodes can connect to each other, otherwise the matrix 
            represents a binary (two-type node) network where only pairs of nodes 
            of different types can connect, where nodes from one type are in the
            rows and from the other type in the columns.

        Returns
        -------
        float :
            BDM value change.
#
        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> perturbation.perturb((10, ), -1) # doctest: +FLOAT_CMP
        26.91763012739709
        """
        #ToDo checks
        
        return self._method(idx, axis, keep_changes, binary_network)

    def run(self, idx=None, axis=0, keep_changes=False, binary_network=False):
        """Run node perturbation experiment.

        Parameters
        ----------
        idx : array_like or None
            *Numpy* integer array providing indices of nodes
            to perturb. If ``None`` then all elements are perturbed.
        axis : array_like or None
            Value to assign during perturbation.
            Negative values correspond to changing value to other
            randomly selected symbols from the alphabet.
            If ``None`` then all values are assigned this way.
            If set then its dimensions must agree with the dimensions
            of ``idx`` (they are horizontally stacked).
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        array_like
            1D float array with perturbation values.

        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> changes = np.array([10, 20])
        >>> perturbation.run(changes) # doctest: +FLOAT_CMP
        array([26.91763013, 27.34823681])
        """
        if idx is None:
            indexes = [ range(k) for k in self.X.shape ]
            idx = np.array([ x for x in product(*indexes) ], dtype=int)
        if values is None:
            values = np.full((idx.shape[0], ), -1, dtype=int)
        return np.apply_along_axis(
            lambda r: self.perturb(tuple(r[:-1]), r[-1], keep_changes=keep_changes),
            axis=1,
            arr=np.column_stack((idx, values))
        )
