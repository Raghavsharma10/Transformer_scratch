def num_samples(self, **kwargs):
        """
        Get the number of samples in this data source.

        Returns
        -------
        int, `np.inf` or `None`.
            An int if the number of samples is known, `np.inf` if it is
            infinite or `None` if the number of samples is unknown.
        """
        if self.num_samples_fn is None:
            return None
        elif callable(self.num_samples_fn):
            return self.num_samples_fn(**kwargs)
        else:
            return self.num_samples_fn