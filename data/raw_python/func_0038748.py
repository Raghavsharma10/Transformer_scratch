def frequency(self, mapping):
        """
        Returns frequency of a given :class:`caspo.core.mapping.Mapping`

        Parameters
        ----------
        mapping : :class:`caspo.core.mapping.Mapping`
            A logical conjuntion mapping

        Returns
        -------
        float
            Frequency of the given mapping over all logical networks

        Raises
        ------
        ValueError
            If the given mapping is not found in the mappings of the underlying hypergraph of this list
        """
        return self.__matrix[:, self.hg.mappings[mapping]].mean()