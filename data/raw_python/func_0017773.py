def class_balancing_sampler(y, indices):
        """
        Construct a `WeightedSubsetSampler` that compensates for class
        imbalance.

        Parameters
        ----------
        y: NumPy array, 1D dtype=int
            sample classes, values must be 0 or positive
        indices: NumPy array, 1D dtype=int
            An array of indices that identify the subset of samples drawn
            from data that are to be used

        Returns
        -------
        WeightedSubsetSampler instance
            Sampler
        """
        weights = WeightedSampler.class_balancing_sample_weights(y[indices])
        return WeightedSubsetSampler(weights, indices=indices)