"""_Algorithms based on ``BDM`` objects."""
from itertools import product
from random import choice
import numpy as np


class NodePerturbationExperiment:
    """Node Perturbation experiment class.

    Node perturbation experiment studies change of BDM / entropy of a 
    network (given their adjacency matrix) when existing nodes are removed.
    This is the main tool for detecting nodes of a network having some 
    causal significance as opposed to noise parts.

    Nodes which when removed yield negative contribution to the overall
    complexity  are likely to be important for the system, since their 
    removal make it more noisy. On the other hand nodes that yield positive 
    contribution to the overall complexity after removal are likely to be noise 
    since they elongate the system's description length.

    Attributes
    ----------
    bdm : BDM
        BDM object. It has to be configured properly to handle
        the dataset that is to be studied.
    X : array_like (optional)
        Dataset for perturbation analysis. May be set later.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.
    bipartite_network: boolean
        If ``False``, the dataset (X) is the adjacency matrix of a one-type node
        network where all nodes can connect to each other, otherwise the matrix 
        represents a bipartite (two-type node) network where only pairs of nodes 
        of different types can connect, where nodes from one category are in the
        rows and nodes from the other category in the columns.

    See also
    --------
    pybdm.bdm.BDM : BDM computations

    Examples
    --------
    >>> import numpy as np
    >>> from pybdm import BDM, NodePerturbationExperiment
    >>> X = np.random.randint(0, 2, (100, 100))
    >>> bdm = BDM(ndim=2)
    >>> npe = NodePerturbationExperiment(bdm, metric='bdm')
    >>> npe.set_data(X)

    >>> idx = [1,5,10,50] # Remove these specific nodes
    >>> delta_bdm = npe.run(idx)
    >>> len(delta_bdm) == len(idx)
    True

    More examples can be found in :doc:`usage`.
    """
    def __init__(self, bdm, X=None, metric='bdm', bipartite_network=False):
        """Initialization method."""
        self.bdm = bdm
        self.metric = metric
        self.bipartite_network = bipartite_network
        self._value = None
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
        if self.metric == 'bdm':
            self._value = self.bdm.bdm(self.X)
        elif self.metric == 'ent':
            self._value = self.bdm.ent(self.X)
        if not self.bipartite_network and self.shape[0]!=self.shape[1]:
            raise ValueError("'X' has to be a squared matrix for non-bipartite network")
            

    def _update_bdm(self, idx, axis, keep_changes, bipartite_network):
        old_bdm = self._value
        if not bipartite_network:
            newX = np.delete(self.X,idx,axis=0)
            newX = np.delete(newX,idx,axis=1)
        else:
            newX = np.delete(self.X,idx,axis=axis)
        new_bdm = self.bdm.bdm(newX)
        if keep_changes:
            self.X = newX
            self._value = new_bdm
        return new_bdm - old_bdm

    def _update_ent(self, idx, axis, keep_changes, bipartite_network):
        old_ent = self._value
        if not bipartite_network:
            newX = np.delete(self.X,idx,axis=0)
            newX = np.delete(newX,idx,axis=1)
        else:
            newX = np.delete(self.X,idx,axis=axis)
        new_ent = self.bdm.ent(newX)
        if keep_changes:
            self.X = newX
            self._value = new_ent
        return new_ent - old_ent

    def perturb(self, idx, axis=0, keep_changes=False):
        """Delete node of the dataset.

        Parameters
        ----------
        idx : int
            Number of row or column of node in adyacency matrix.
        axis : int
            If bipartite_network is ``True``, is the axis that is used to remove node,
            otherwise is ignored.
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        float :
            BDM value change.

        Examples
        --------
        >>> import numpy as np
        >>> from pybdm import BDM, NodePerturbationExperiment
        >>> bdm = BDM(ndim=2)
        >>> X = np.random.randint(0, 2, (100, 100))
        >>> perturbation = NodePerturbationExperiment(bdm, X)
        >>> perturbation.perturb(7)
        -1541.9845807106612
        """
        
        return self._method(idx, axis, keep_changes,
                            bipartite_network=self.bipartite_network)

    def run(self, first_idx=None, second_idx=None, axis=0):
        """Run node perturbation experiment. Calls the function self.perturb for
        each node index and keep_changes=False.

        Parameters
        ----------
        first_idx : an array of row indices to be perturbed in the adyacency matrix.
            If the network is not bipartite, is the index for both rows and columns, 
            otherwise

        axis : array_like or None
            Value to assign during perturbation.
            Negative values correspond to changing value to other
            randomly selected symbols from the alphabet.
            If ``None`` then all values are assigned this way.
            If set then its dimensions must agree with the dimensions
            of ``idx`` (they are horizontally stacked).

        Returns
        -------
        For bipartite_network==True: 
            One 1D float array with perturbation values for each node.
        For bipartite_network==False:
            Two 1D float arrays with perturbation values corresponding to row nodes
            and column nodes.

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
        if first_idx is None and second_idx is not None:
            raise ValueError("There needs to be a value for first_idx if a value for second_idx is supplied")
        
        if not self.bipartite_network:
            if first_idx is None:
                first_idx = np.arange(self.shape[0])
            output = np.array([self._method(x,axis=axis, keep_changes=False,
                                            bipartite_network=self.bipartite_network)
                               for x in first_idx])
            return output
        else:
            if first_idx is not None and second_idx is None:
                output = np.array([self._method(x, axis=axis,keep_changes=False,
                                                bipartite_network=self.bipartite_network)
                                   for x in first_idx])
                return output
            if first_idx is None and second_idx is None:
                first_idx = np.arange(self.shape[0])
                second_idx = np.arange(self.shape[1])
            out_rows = np.array([self._method(x, axis=0,keep_changes=False,
                                              bipartite_network=self.bipartite_network)
                                 for x in first_idx])
            out_cols = np.array([self._method(x, axis=1,keep_changes=False,
                                              bipartite_network=self.bipartite_network)
                                 for x in second_idx])
            return out_rows, out_cols
 
                
