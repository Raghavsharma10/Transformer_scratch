def to_grey(self, on: bool=False):
        """
        Change the LED to grey.

        :param on: Unused, here for API consistency with the other states
        :return: None
        """
        self._on = False
        self._load_new(led_grey)