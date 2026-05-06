def set_window_override_redirect(self, window, override_redirect):
        """
        Set the override_redirect value for a window. This generally means
        whether or not a window manager will manage this window.

        If you set it to 1, the window manager will usually not draw
        borders on the window, etc. If you set it to 0, the window manager
         will see it like a normal application window.
        """
        _libxdo.xdo_set_window_override_redirect(
            self._xdo, window, override_redirect)