def to_red(self, on: bool=False):
        """
        Change the LED to red (on or off)
        :param on: True or False
        :return: None
        """
        self._on = on
        if on:
            self._load_new(led_red_on)

            if self._toggle_on_click:
                self._canvas.bind('<Button-1>', lambda x: self.to_red(False))
        else:
            self._load_new(led_red)

            if self._toggle_on_click:
                self._canvas.bind('<Button-1>', lambda x: self.to_red(True))