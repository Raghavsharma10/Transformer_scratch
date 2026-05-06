def set_window_urgency(self, window, urgency):
        """Sets the urgency hint for a window"""
        _libxdo.xdo_set_window_urgency(self._xdo, window, urgency)