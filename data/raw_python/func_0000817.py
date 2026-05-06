def send_keysequence_window_down(self, window, keysequence, delay=12000):
        """Send key press (down) events for the given key sequence"""
        _libxdo.xdo_send_keysequence_window_down(
            self._xdo, window, keysequence, ctypes.c_ulong(delay))