def wait_for_mouse_move_from(self, origin_x, origin_y):
        """
        Wait for the mouse to move from a location. This function will block
        until the condition has been satisified.

        :param origin_x: the X position you expect the mouse to move from
        :param origin_y: the Y position you expect the mouse to move from
        """
        _libxdo.xdo_wait_for_mouse_move_from(self._xdo, origin_x, origin_y)