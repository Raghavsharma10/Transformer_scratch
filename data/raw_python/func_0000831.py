def wait_for_window_active(self, window, active=1):
        """
        Wait for a window to be active or not active.

        Requires your window manager to support this.
        Uses _NET_ACTIVE_WINDOW from the EWMH spec.

        :param window: the window to wait on
        :param active: If 1, wait for active. If 0, wait for inactive.
        """
        _libxdo.xdo_wait_for_window_active(self._xdo, window, active)