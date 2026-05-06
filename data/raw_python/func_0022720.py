def width_max(self, width_max):
        """Set the maximum width of the widget.

        Parameters
        ----------
        width_max: None | float
            the maximum width of the widget. if None, maximum width
            is unbounded
        """
        if width_max is None:
            self._width_limits[1] = None
            return

        width_max = float(width_max)
        assert(self.width_min <= width_max)

        self._width_limits[1] = width_max
        self._update_layout()