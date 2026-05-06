def enter_text_window(self, window, string, delay=12000):
        """
        Type a string to the specified window.

        If you want to send a specific key or key sequence, such as
        "alt+l", you want instead ``send_keysequence_window(...)``.

        :param window:
            The window you want to send keystrokes to or CURRENTWINDOW
        :param string:
            The string to type, like "Hello world!"
        :param delay:
            The delay between keystrokes in microseconds.
            12000 is a decent choice if you don't have other plans.
        """
        return _libxdo.xdo_enter_text_window(self._xdo, window, string, delay)