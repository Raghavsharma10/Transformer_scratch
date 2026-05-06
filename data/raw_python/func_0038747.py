def frequencies_iter(self):
        """
        Iterates over all non-zero frequencies of logical conjunction mappings in this list

        Yields
        ------
        tuple[caspo.core.mapping.Mapping, float]
            The next pair (mapping,frequency)
        """
        f = self.__matrix.mean(axis=0)
        for i, m in self.mappings.iteritems():
            yield m, f[i]