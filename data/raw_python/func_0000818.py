def send_keysequence_window_list_do(
            self, window, keys, pressed=1, modifier=None, delay=120000):
        """
        Send a series of keystrokes.

        :param window: The window to send events to or CURRENTWINDOW
        :param keys: The array of charcodemap_t entities to send.
        :param pressed: 1 for key press, 0 for key release.
        :param modifier:
            Pointer to integer to record the modifiers
            activated by the keys being pressed. If NULL, we don't save
            the modifiers.
        :param delay:
            The delay between keystrokes in microseconds.
        """
        # todo: how to properly use charcodes_t in a nice way?
        _libxdo.xdo_send_keysequence_window_list_do(
            self._xdo, window, keys, len(keys), pressed, modifier, delay)