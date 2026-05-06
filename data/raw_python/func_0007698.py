def to_green(self, on: bool=False):
        """
        Change the LED to green (on or off).

        :param on: True or False
        :return: None
        """
        self._on = on
        if on:
            self._load_new(led_green_on)

            if self._toggle_on_click:
                self._canvas.bind('<Button-1>', lambda x: self.to_green(False))
        else:
            self._load_new(led_green)

            if self._toggle_on_click:
                self._canvas.bind('<Button-1>', lambda x: self.to_green(True))