def batch_indices_iterator(self, batch_size, shuffle=None, **kwargs):
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
        shuffle: `numpy.random.RandomState` or `True` or `None`
            Used to randomise element order. If `None`, elements will be
            extracted in order. If it is a `RandomState` instance, that
            RNG will be used to shuffle elements. If it is `True`, NumPy's
            default RNG will be used.

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
        shuffle_rng = self._get_shuffle_rng(shuffle)
        iterators = [d.batch_indices_iterator(batch_size,
                                              shuffle=shuffle_rng, **kwargs)
                     for d in self.datasets]
        return self._ds_iterator(batch_size, iterators, shuffle_rng, **kwargs)