def get_active_window(self):
        """
        Get the currently-active window.
        Requires your window manager to support this.
        Uses ``_NET_ACTIVE_WINDOW`` from the EWMH spec.
        """
        window_ret = window_t(0)
        _libxdo.xdo_get_active_window(self._xdo, ctypes.byref(window_ret))
        return window_ret.value