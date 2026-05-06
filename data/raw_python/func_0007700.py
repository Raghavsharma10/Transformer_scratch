def to_yellow(self, on: bool=False):
        """
        Change the LED to yellow (on or off)
        :param on: True or False
        :return: None
        """
        self._on = on
        if on:
            self._load_new(led_yellow_on)

            if self._toggle_on_click:
                self._canvas.bind('<Button-1>',
                                  lambda x: self.to_yellow(False))
        else:
            self._load_new(led_yellow)

            if self._toggle_on_click:
                self._canvas.bind('<Button-1>',
                                  lambda x: self.to_yellow(True))