def select_window_with_click(self):
        """
        Get a window ID by clicking on it.
        This function blocks until a selection is made.
        """
        window_ret = window_t(0)
        _libxdo.xdo_select_window_with_click(
            self._xdo, ctypes.byref(window_ret))
        return window_ret.value