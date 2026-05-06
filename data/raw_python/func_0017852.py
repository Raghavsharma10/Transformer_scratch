def samples_by_indices(self, indices):
        """
        Gather a batch of samples by indices, applying any index
        mapping defined by the underlying data sources.

        Parameters
        ----------
        indices: 1D-array of ints or slice
            An index array or a slice that selects the samples to retrieve

        Returns
        -------
        nested list of arrays
            A mini-batch
        """
        if not self._random_access:
            raise TypeError('samples_by_indices method not supported as one '
                            'or more of the underlying data sources does '
                            'not support random access')
        batch = self.source.samples_by_indices(indices)
        return self.fn(*batch)