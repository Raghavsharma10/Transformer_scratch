def calculate_colorbar(self):
        """
        Returns the positions and colors of all intervals inside the colorbar.
        """
        self._base._process_values()
        self._base._find_range()
        X, Y = self._base._mesh()
        C = self._base._values[:, np.newaxis]
        return X, Y, C