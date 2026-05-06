def batch_indices_iterator(self, batch_size, **kwargs):
        """
        Create an iterator that generates mini-batch sample indices

        The generated mini-batches indices take the form of nested lists of
        either:
        - 1D NumPy integer arrays
        - slices

        The list nesting structure with match that of the tree of data sources
        rooted at `self`

        Parameters
        ----------
        batch_size: int
            Mini-batch size

        Returns
        -------
        iterator
            An iterator that generates items that are nested lists of slices
            or 1D NumPy integer arrays.
        """
        if not self._random_access:
            raise TypeError('batch_indices_iterator method not supported as '
                            'one or more of the underlying data sources '
                            'does not support random access')
        iterators = [d.batch_indices_iterator(batch_size, **kwargs)
                     for d in self.datasets]

        for batch in six.moves.zip(*iterators):
            yield self._prepare_index_batch(batch)