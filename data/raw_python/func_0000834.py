def get_window_size(self, window):
        """
        Get a window's size.
        """
        w_ret = ctypes.c_uint(0)
        h_ret = ctypes.c_uint(0)
        _libxdo.xdo_get_window_size(self._xdo, window, ctypes.byref(w_ret),
                                    ctypes.byref(h_ret))
        return window_size(w_ret.value, h_ret.value)