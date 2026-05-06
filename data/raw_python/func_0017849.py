def samples_by_indices_nomapping(self, indices):
        """
        Gather a batch of samples by indices *without* applying any index
        mapping.

        Parameters
        ----------
        indices: a tuple of the form `(dataset_index, sample_indices)`
            The `dataset_index` identifies the dataset from which to draw
            samples while `sample_indices` identifies the samples to draw
            from it.

        Returns
        -------
        nested list of arrays
            A mini-batch
        """
        if not self._random_access:
            raise TypeError('samples_by_indices_nomapping method not '
                            'supported as one or more of the underlying '
                            'data sources does not support random access')
        if not isinstance(indices, tuple):
            raise TypeError('indices should be a tuple, not a {}'.format(
                type(indices)
            ))
        dataset_index, sample_indices = indices
        ds = self.datasets[dataset_index]
        return ds.samples_by_indices_nomapping(sample_indices)