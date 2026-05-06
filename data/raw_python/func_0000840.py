def get_desktop_for_window(self, window):
        """
        Get the desktop a window is on.
        Uses _NET_WM_DESKTOP of the EWMH spec.

        If your desktop does not support ``_NET_WM_DESKTOP``, then '*desktop'
        remains unmodified.

        :param wid: the window to query
        """
        desktop = ctypes.c_long(0)
        _libxdo.xdo_get_desktop_for_window(
            self._xdo, window, ctypes.byref(desktop))
        return desktop.value