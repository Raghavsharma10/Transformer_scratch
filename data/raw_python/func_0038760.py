def to_array(self, mappings):
        """
        Converts the logical network to a binary array with respect to the given mappings from a
        :class:`caspo.core.hypergraph.HyperGraph` object instance.

        Parameters
        ----------
        mappings : :class:`caspo.core.mapping.MappingList`
            Mappings to create the binary array

        Returns
        -------
        `numpy.ndarray`_
            Binary array with respect to the given mappings describing the logical network.
            Position `i` in the array will be 1 if the network has the mapping at position `i`
            in the given list of mappings.


        .. _numpy.ndarray: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray
        """
        arr = np.zeros(len(mappings), np.int8)
        for i, (clause, target) in enumerate(mappings):
            if self.has_edge(clause, target):
                arr[i] = 1

        return arr