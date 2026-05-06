def from_hypertuples(cls, hg, tuples):
        """
        Creates a logical network from an iterable of integer tuples matching mappings in the given
        :class:`caspo.core.hypergraph.HyperGraph`

        Parameters
        ----------
        hg : :class:`caspo.core.hypergraph.HyperGraph`
            Underlying hypergraph

        tuples : (int,int)
            tuples matching mappings in the given hypergraph

        Returns
        -------
        caspo.core.logicalnetwork.LogicalNetwork
            Created object instance
        """
        return cls([(hg.clauses[j], hg.variable(i)) for i, j in tuples], networks=1)