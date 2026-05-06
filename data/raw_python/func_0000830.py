def get_focused_window_sane(self):
        """
        Like xdo_get_focused_window, but return the first ancestor-or-self
        window * having a property of WM_CLASS. This allows you to get
        the "real" or top-level-ish window having focus rather than something
        you may not expect to be the window having focused.

        :param window_ret:
            Pointer to a window where the currently-focused window
            will be stored.
        """
        window_ret = window_t(0)
        _libxdo.xdo_get_focused_window_sane(
            self._xdo, ctypes.byref(window_ret))
        return window_ret.value