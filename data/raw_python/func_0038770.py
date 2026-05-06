def combinatorics(self):
        """
        Returns mutually exclusive/inclusive clampings

        Returns
        -------
        (dict,dict)
            A tuple of 2 dictionaries.
            For each literal key, the first dict has as value the set of mutually exclusive clampings while
            the second dict has as value the set of mutually inclusive clampings.
        """
        df = self.to_dataframe()
        literals = set((l for l in it.chain.from_iterable(self)))
        exclusive, inclusive = defaultdict(set), defaultdict(set)

        for l1, l2 in it.combinations(it.ifilter(lambda l: self.frequency(l) < 1., literals), 2):
            a1, a2 = df[l1.variable] == l1.signature, df[l2.variable] == l2.signature
            if (a1 != a2).all():
                exclusive[l1].add(l2)
                exclusive[l2].add(l1)

            if (a1 == a2).all():
                inclusive[l1].add(l2)
                inclusive[l2].add(l1)

        return exclusive, inclusive