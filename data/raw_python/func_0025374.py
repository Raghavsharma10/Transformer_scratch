def period_max_neighborhood_probability(self, threshold, radius, sigmas=None):
        """
        Calculates the neighborhood probability of exceeding a threshold at any time over the period loaded.

        Args:
            threshold (float): splitting threshold for probability calculatations
            radius (int): distance from point in number of grid points to include in neighborhood calculation.
            sigmas (array of ints): Radii for Gaussian filter used to smooth neighborhood probabilities.

        Returns:
            list of EnsembleConsensus objects
        """
        if sigmas is None:
            sigmas = [0]
        weights = disk(radius)
        neighborhood_prob = np.zeros(self.data.shape[2:], dtype=np.float32)
        thresh_data = np.zeros(self.data.shape[2:], dtype=np.uint8)
        for m in range(self.data.shape[0]):
            thresh_data[self.data[m].max(axis=0) >= threshold] = 1
            maximized = fftconvolve(thresh_data, weights, mode="same")
            maximized[maximized > 1] = 1
            neighborhood_prob += fftconvolve(maximized, weights, mode="same")
        neighborhood_prob[neighborhood_prob < 1] = 0
        neighborhood_prob /= (self.data.shape[0] * float(weights.sum()))
        consensus_probs = []
        for sigma in sigmas:
            if sigma > 0:
                filtered_prob = gaussian_filter(neighborhood_prob, sigma=sigma)
            else:
                filtered_prob = neighborhood_prob
            ec = EnsembleConsensus(filtered_prob,
                                   "neighbor_prob_{0:02d}-hour_r_{1:d}_s_{2:d}".format(self.data.shape[1],
                                                                                       radius, sigma),
                                   self.ensemble_name,
                                   self.run_date, self.variable + "_{0:0.2f}".format(float(threshold)),
                                   self.start_date, self.end_date, "")
            consensus_probs.append(ec)
        return consensus_probs