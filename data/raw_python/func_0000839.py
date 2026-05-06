def set_desktop_for_window(self, window, desktop):
        """
        Move a window to another desktop
        Uses _NET_WM_DESKTOP of the EWMH spec.

        :param wid: the window to move
        :param desktop: the desktop destination for the window
        """
        _libxdo.xdo_set_desktop_for_window(self._xdo, window, desktop)