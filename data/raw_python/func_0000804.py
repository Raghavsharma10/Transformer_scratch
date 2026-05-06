def move_mouse_relative(self, x, y):
        """
        Move the mouse relative to it's current position.

        :param x: the distance in pixels to move on the X axis.
        :param y: the distance in pixels to move on the Y axis.
        """
        _libxdo.xdo_move_mouse_relative(self._xdo, x, y)