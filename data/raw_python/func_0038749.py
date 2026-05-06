def combinatorics(self):
        """
        Returns mutually exclusive/inclusive mappings

        Returns
        -------
        (dict,dict)
            A tuple of 2 dictionaries.
            For each mapping key, the first dict has as value the set of mutually exclusive mappings while
            the second dict has as value the set of mutually inclusive mappings.
        """
        f = self.__matrix.mean(axis=0)
        candidates = np.where((f < 1) & (f > 0))[0]
        exclusive, inclusive = defaultdict(set), defaultdict(set)
        for i, j in it.combinations(candidates, 2):
            xor = np.logical_xor(self.__matrix[:, i], self.__matrix[:, j])
            if xor.all():
                exclusive[self.hg.mappings[i]].add(self.hg.mappings[j])
                exclusive[self.hg.mappings[j]].add(self.hg.mappings[i])

            if (~xor).all():
                inclusive[self.hg.mappings[i]].add(self.hg.mappings[j])
                inclusive[self.hg.mappings[j]].add(self.hg.mappings[i])

        return exclusive, inclusive