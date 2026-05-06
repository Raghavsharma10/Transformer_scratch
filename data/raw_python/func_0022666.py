def set_color(self, color):
        """Set the color

        Parameters
        ----------
        color : instance of Color
            The color to use.
        """
        if color is not None:
            self._color_lev = color
            self._need_color_update = True
            self.update()