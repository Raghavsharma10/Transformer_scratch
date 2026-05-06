def get_by_index(self, i):
        """Look up a gene set by its index.

        Parameters
        ----------
        i: int
            The index of the gene set.

        Returns
        -------
        GeneSet
            The gene set.

        Raises
        ------
        ValueError
            If the given index is out of bounds.
        """
        if i >= self.n:
            raise ValueError('Index %d out of bounds ' % i +
                             'for database with %d gene sets.' % self.n)
        return self._gene_sets[self._gene_set_ids[i]]