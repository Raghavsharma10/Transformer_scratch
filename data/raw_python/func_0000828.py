def get_focused_window(self):
        """
        Get the window currently having focus.

        :param window_ret:
        Pointer to a window where the currently-focused window
        will be stored.
        """
        window_ret = window_t(0)
        _libxdo.xdo_get_focused_window(self._xdo, ctypes.byref(window_ret))
        return window_ret.value