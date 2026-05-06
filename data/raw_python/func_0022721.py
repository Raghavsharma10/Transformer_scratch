def height_min(self, height_min):
        """Set the minimum height of the widget

        Parameters
        ----------

        height_min: float
            the minimum height of the widget
        """
        if height_min is None:
            self._height_limits[0] = 0
            return

        height_min = float(height_min)
        assert(height_min >= 0)

        self._height_limits[0] = height_min
        self._update_layout()