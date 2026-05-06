def from_hypergraph(cls, hypergraph, networks=None):
        """
        Creates a list of logical networks from a given hypergraph and an
        optional list of :class:`caspo.core.logicalnetwork.LogicalNetwork` object instances

        Parameters
        ----------
        hypegraph : :class:`caspo.core.hypergraph.HyperGraph`
            Underlying hypergraph for this logical network list

        networks : Optional[list]
            List of :class:`caspo.core.logicalnetwork.LogicalNetwork` object instances

        Returns
        -------
        caspo.core.logicalnetwork.LogicalNetworkList
           Created object instance
        """
        matrix = None
        nnet = None
        if networks:
            matrix = np.array([networks[0].to_array(hypergraph.mappings)])
            nnet = [networks[0].networks]
            for network in networks[1:]:
                matrix = np.append(matrix, [network.to_array(hypergraph.mappings)], axis=0)
                nnet.append(network.networks)

        return cls(hypergraph, matrix, nnet)