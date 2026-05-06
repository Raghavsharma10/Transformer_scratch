def send_keysequence_window_up(self, window, keysequence, delay=12000):
        """Send key release (up) events for the given key sequence"""
        _libxdo.xdo_send_keysequence_window_up(
            self._xdo, window, keysequence, ctypes.c_ulong(delay))