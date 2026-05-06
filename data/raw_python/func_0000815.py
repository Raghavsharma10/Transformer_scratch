def send_keysequence_window(self, window, keysequence, delay=12000):
        """
        Send a keysequence to the specified window.

        This allows you to send keysequences by symbol name. Any combination
        of X11 KeySym names separated by '+' are valid. Single KeySym names
        are valid, too.

        Examples:
          "l"
          "semicolon"
          "alt+Return"
          "Alt_L+Tab"

        If you want to type a string, such as "Hello world." you want to
        instead use xdo_enter_text_window.

        :param window: The window you want to send the keysequence to or
          CURRENTWINDOW
        :param keysequence: The string keysequence to send.
        :param delay: The delay between keystrokes in microseconds.
        """
        _libxdo.xdo_send_keysequence_window(
            self._xdo, window, keysequence, delay)