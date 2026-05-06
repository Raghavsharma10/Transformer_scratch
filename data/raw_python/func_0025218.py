def quantize(self, input_grid):
        """
        Quantize a grid into discrete steps based on input parameters.

        Args:
            input_grid: 2-d array of values

        Returns:
            Dictionary of value pointing to pixel locations, and quantized 2-d array of data
        """
        pixels = {}
        for i in range(self.max_bin+1):
            pixels[i] = []

        data = (np.array(input_grid, dtype=int) - self.min_thresh) / self.data_increment
        data[data < 0] = -1
        data[data > self.max_bin] = self.max_bin
        good_points = np.where(data >= 0)
        for g in np.arange(good_points[0].shape[0]):
            pixels[data[(good_points[0][g], good_points[1][g])]].append((good_points[0][g], good_points[1][g]))
        return pixels, data