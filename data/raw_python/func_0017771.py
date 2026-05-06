def class_balancing_sample_weights(y):
        """
        Compute sample weight given an array of sample classes. The weights
        are assigned on a per-class basis and the per-class weights are
        inversely proportional to their frequency.

        Parameters
        ----------
        y: NumPy array, 1D dtype=int
            sample classes, values must be 0 or positive

        Returns
        -------
        NumPy array, 1D dtype=float
            per sample weight array
        """
        h = np.bincount(y)
        cls_weight = 1.0 / (h.astype(float) * len(np.nonzero(h)[0]))
        cls_weight[np.isnan(cls_weight)] = 0.0
        sample_weight = cls_weight[y]
        return sample_weight