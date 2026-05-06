def concat(self, other):
        """
        Returns the concatenation with another :class:`caspo.core.logicalnetwork.LogicalNetworkList` object instance.
        It is assumed (not checked) that both have the same underlying hypergraph.

        Parameters
        ----------
        other : :class:`caspo.core.logicalnetwork.LogicalNetworkList`
            The list to concatenate

        Returns
        -------
        caspo.core.logicalnetwork.LogicalNetworkList
            If other is empty returns self, if self is empty returns other, otherwise a new
            :class:`caspo.core.LogicalNetworkList` is created by concatenating self and other.
        """
        if len(other) == 0:
            return self
        elif len(self) == 0:
            return other
        else:
            return LogicalNetworkList(self.hg, np.append(self.__matrix, other.__matrix, axis=0), np.concatenate([self.__networks, other.__networks]))