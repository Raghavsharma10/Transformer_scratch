def get_current_desktop(self):
        """
        Get the current desktop.
        Uses ``_NET_CURRENT_DESKTOP`` of the EWMH spec.
        """
        desktop = ctypes.c_long(0)
        _libxdo.xdo_get_current_desktop(self._xdo, ctypes.byref(desktop))
        return desktop.value