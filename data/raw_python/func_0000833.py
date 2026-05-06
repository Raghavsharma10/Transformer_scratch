def get_window_location(self, window):
        """
        Get a window's location.
        """
        screen_ret = Screen()
        x_ret = ctypes.c_int(0)
        y_ret = ctypes.c_int(0)
        _libxdo.xdo_get_window_location(
            self._xdo, window, ctypes.byref(x_ret), ctypes.byref(y_ret),
            ctypes.byref(screen_ret))
        return window_location(x_ret.value, y_ret.value, screen_ret)