def mouse_down(self, window, button):
        """
        Send a mouse press (aka mouse down) for a given button at
        the current mouse location.

        :param window:
            The window you want to send the event to or CURRENTWINDOW
        :param button:
            The mouse button. Generally, 1 is left, 2 is middle, 3 is
            right, 4 is wheel up, 5 is wheel down.
        """
        _libxdo.xdo_mouse_down(
            self._xdo, ctypes.c_ulong(window), ctypes.c_int(button))