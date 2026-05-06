def wait_for_mouse_move_to(self, dest_x, dest_y):
        """
        Wait for the mouse to move to a location. This function will block
        until the condition has been satisified.

        :param dest_x: the X position you expect the mouse to move to
        :param dest_y: the Y position you expect the mouse to move to
        """
        _libxdo.xdo_wait_for_mouse_move_from(self._xdo, dest_x, dest_y)