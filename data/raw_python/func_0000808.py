def get_window_at_mouse(self):
        """
        Get the window the mouse is currently over
        """
        window_ret = ctypes.c_ulong(0)
        _libxdo.xdo_get_window_at_mouse(self._xdo, ctypes.byref(window_ret))
        return window_ret.value