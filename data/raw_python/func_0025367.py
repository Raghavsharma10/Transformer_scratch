def neighborhood_probability(self, threshold, radius):
        """
        Calculate a probability based on the number of grid points in an area that exceed a threshold.

        Args:
            threshold:
            radius:

        Returns:

        """
        weights = disk(radius, dtype=np.uint8)
        thresh_data = np.zeros(self.data.shape[1:], dtype=np.uint8)
        neighbor_prob = np.zeros(self.data.shape, dtype=np.float32)
        for t in np.arange(self.data.shape[0]):
            thresh_data[self.data[t] >= threshold] = 1
            maximized = fftconvolve(thresh_data, weights, mode="same")
            maximized[maximized > 1] = 1
            maximized[maximized < 1] = 0
            neighbor_prob[t] = fftconvolve(maximized, weights, mode="same")
            thresh_data[:] = 0
        neighbor_prob[neighbor_prob < 1] = 0
        neighbor_prob /= weights.sum()
        return neighbor_prob