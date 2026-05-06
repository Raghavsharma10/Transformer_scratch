def height_max(self, height_max):
        """Set the maximum height of the widget.

        Parameters
        ----------
        height_max: None | float
            the maximum height of the widget. if None, maximum height
            is unbounded
        """
        if height_max is None:
            self._height_limits[1] = None
            return

        height_max = float(height_max)
        assert(0 <= self.height_min <= height_max)
        self._height_limits[1] = height_max
        self._update_layout()