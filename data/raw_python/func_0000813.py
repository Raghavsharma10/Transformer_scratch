def click_window_multiple(self, window, button, repeat=2, delay=100000):
        """
        Send a one or more clicks for a specific mouse button at the
        current mouse location.

        :param window:
            The window you want to send the event to or CURRENTWINDOW
        :param button:
            The mouse button. Generally, 1 is left, 2 is middle, 3 is
            right, 4 is wheel up, 5 is wheel down.
        :param repeat: number of repetitions (default: 2)
        :param delay: delay between clicks, in microseconds (default: 100k)
        """
        _libxdo.xdo_click_window_multiple(
            self._xdo, window, button, repeat, delay)