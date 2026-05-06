def wait_for_window_focus(self, window, want_focus):
        """
        Wait for a window to have or lose focus.

        :param window: The window to wait on
        :param want_focus: If 1, wait for focus. If 0, wait for loss of focus.
        """
        _libxdo.xdo_wait_for_window_focus(self._xdo, window, want_focus)