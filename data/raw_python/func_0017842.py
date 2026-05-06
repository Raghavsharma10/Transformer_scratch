def samples_by_indices(self, indices):
        """
        Gather a batch of samples by indices, applying the mapping
        described by the (optional) `indices` array passed to the
        constructor.

        Parameters
        ----------
        indices: 1D-array of ints or slice
            The samples to retrieve

        Returns
        -------
        list of arrays
            A mini-batch in the form of a list of NumPy arrays
        """
        indices = self.sampler.map_indices(indices)
        return self.samples_by_indices_nomapping(indices)