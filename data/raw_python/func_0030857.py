def sort_samples(self, stable=True, inplace=False, ascending=True):
        """Sort the columns of the matrix alphabetically by sample name.

        Parameters
        ----------
        stable: bool, optional
            Whether to use a stable sorting algorithm. [True]
        inplace: bool, optional
            Whether to perform the operation in place.[False]
        ascending: bool, optional
            Whether to sort in ascending order [True]

        Returns
        -------
        `ExpMatrix`
            The sorted matrix.
        """
        kind = 'quicksort'
        if stable:
            kind = 'mergesort'
        return self.sort_index(axis=1, kind=kind, inplace=inplace,
                               ascending=ascending)