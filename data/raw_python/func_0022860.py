def histogram(self, data, bins=10, color='w', orientation='h'):
        """Calculate and show a histogram of data

        Parameters
        ----------
        data : array-like
            Data to histogram. Currently only 1D data is supported.
        bins : int | array-like
            Number of bins, or bin edges.
        color : instance of Color
            Color of the histogram.
        orientation : {'h', 'v'}
            Orientation of the histogram.

        Returns
        -------
        hist : instance of Polygon
            The histogram polygon.
        """
        self._configure_2d()
        hist = scene.Histogram(data, bins, color, orientation)
        self.view.add(hist)
        self.view.camera.set_range()
        return hist