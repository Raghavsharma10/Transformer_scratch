def samples_by_indices_nomapping(self, indices):
        """
        Gather a batch of samples by indices *without* applying any index
        mapping.

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
            raise TypeError('samples_by_indices_nomapping method not '
                            'supported as one or more of the underlying '
                            'data sources does not support random access')
        batch = self.source.samples_by_indices_nomapping(indices)
        return self.fn(*batch)