def samples_by_indices_nomapping(self, indices):
        """
        Gather a batch of samples by indices *without* applying any index
        mapping.

        Parameters
        ----------
        indices: list of either 1D-array of ints or slice
            A list of index arrays or slices; one for each data source
            that identify the samples to access

        Returns
        -------
        nested list of arrays
            A mini-batch
        """
        if not self._random_access:
            raise TypeError('samples_by_indices_nomapping method not '
                            'supported as one or more of the underlying '
                            'data sources does not support random access')
        if len(indices) != len(self.datasets):
            raise ValueError(
                'length mis-match: indices has {} items, self has {} data '
                'sources, should be equal'.format(len(indices),
                                                  len(self.datasets)))
        batch = tuple([ds.samples_by_indices_nomapping(ndx)
                       for ds, ndx in zip(self.datasets, indices)])
        return self._prepare_batch(batch)