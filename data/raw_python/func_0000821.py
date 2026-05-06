def move_window(self, window, x, y):
        """
        Move a window to a specific location.

        The top left corner of the window will be moved to the x,y coordinate.

        :param wid: the window to move
        :param x: the X coordinate to move to.
        :param y: the Y coordinate to move to.
        """
        _libxdo.xdo_move_window(self._xdo, window, x, y)