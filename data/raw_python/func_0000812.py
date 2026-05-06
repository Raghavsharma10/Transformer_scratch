def click_window(self, window, button):
        """
        Send a click for a specific mouse button at the current mouse location.

        :param window:
            The window you want to send the event to or CURRENTWINDOW
        :param button:
            The mouse button. Generally, 1 is left, 2 is middle, 3 is
            right, 4 is wheel up, 5 is wheel down.
        """
        _libxdo.xdo_click_window(self._xdo, window, button)