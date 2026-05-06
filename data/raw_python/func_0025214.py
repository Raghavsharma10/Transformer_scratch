def label(self, input_grid):
        """
        Labels input grid using enhanced watershed algorithm.

        Args:
            input_grid (numpy.ndarray): Grid to be labeled.

        Returns:
            Array of labeled pixels
        """
        marked = self.find_local_maxima(input_grid)
        marked = np.where(marked >= 0, 1, 0)
        # splabel returns two things in a tuple: an array and an integer
        # assign the first thing (array) to markers
        markers = splabel(marked)[0]
        return markers