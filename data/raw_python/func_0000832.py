def reparent_window(self, window_source, window_target):
        """
        Reparents a window

        :param wid_source: the window to reparent
        :param wid_target: the new parent window
        """
        _libxdo.xdo_reparent_window(self._xdo, window_source, window_target)