def set_window_size(self, window, w, h, flags=0):
        """
        Change the window size.

        :param wid: the window to resize
        :param w: the new desired width
        :param h: the new desired height
        :param flags: if 0, use pixels for units. If SIZE_USEHINTS, then
            the units will be relative to the window size hints.
        """
        _libxdo.xdo_set_window_size(self._xdo, window, w, h, flags)