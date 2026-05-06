def neighborhood_probability(self, threshold, radius, sigmas=None):
        """
        Hourly probability of exceeding a threshold based on model values within a specified radius of a point.

        Args:
            threshold (float): probability of exceeding this threshold
            radius (int): distance from point in number of grid points to include in neighborhood calculation.
            sigmas (array of ints): Radii for Gaussian filter used to smooth neighborhood probabilities.

        Returns:
            list of EnsembleConsensus objects containing neighborhood probabilities for each forecast hour.
        """
        if sigmas is None:
            sigmas = [0]
        weights = disk(radius)
        filtered_prob = []
        for sigma in sigmas:
            filtered_prob.append(EnsembleConsensus(np.zeros(self.data.shape[1:], dtype=np.float32),
                                                   "neighbor_prob_r_{0:d}_s_{1:d}".format(radius, sigma),
                                                   self.ensemble_name,
                                                   self.run_date, self.variable + "_{0:0.2f}".format(threshold),
                                                   self.start_date, self.end_date, ""))
        thresh_data = np.zeros(self.data.shape[2:], dtype=np.uint8)
        neighbor_prob = np.zeros(self.data.shape[2:], dtype=np.float32)
        for t in range(self.data.shape[1]):
            for m in range(self.data.shape[0]):
                thresh_data[self.data[m, t] >= threshold] = 1
                maximized = fftconvolve(thresh_data, weights, mode="same")
                maximized[maximized > 1] = 1
                maximized[maximized < 1] = 0
                neighbor_prob += fftconvolve(maximized, weights, mode="same")
                neighbor_prob[neighbor_prob < 1] = 0
                thresh_data[:] = 0
            neighbor_prob /= (self.data.shape[0] * float(weights.sum()))
            for s, sigma in enumerate(sigmas):
                if sigma > 0:
                    filtered_prob[s].data[t] = gaussian_filter(neighbor_prob, sigma=sigma)
                else:
                    filtered_prob[s].data[t] = neighbor_prob
            neighbor_prob[:] = 0
        return filtered_prob