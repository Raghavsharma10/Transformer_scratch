def samples_by_indices_nomapping(self, indices):
        """
        Gather a batch of samples by indices *without* applying any index
        mapping resulting from the (optional) use of the `indices` array
        passed to the constructor.

        Parameters
        ----------
        indices: 1D-array of ints or slice
            The samples to retrieve

        Returns
        -------
        list of arrays
            A mini-batch in the form of a list of NumPy arrays
        """
        batch = tuple([d[indices] for d in self.data])
        if self.include_indices:
            if isinstance(indices, slice):
                indices = np.arange(indices.start, indices.stop,
                                    indices.step)
            return (indices,) + batch
        else:
            return batch