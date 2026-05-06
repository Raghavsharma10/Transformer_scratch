def width_min(self, width_min):
        """Set the minimum height of the widget

        Parameters
        ----------

        height_min: float
            the minimum height of the widget
        """

        if width_min is None:
            self._width_limits[0] = 0
            return

        width_min = float(width_min)
        assert(0 <= width_min)

        self._width_limits[0] = width_min
        self._update_layout()