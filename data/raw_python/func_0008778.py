def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, _ in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j