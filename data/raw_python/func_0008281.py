def _pollMouse(self):
        """
        Polls @10Hz, with a slight delay at the
        start.
        """
        if self._mouseJustPressed:
            delay = 300
            self._mouseJustPressed = False
        else:
            delay = 100

        if self._leftMousePressed:
            self.add(1)
            self.after_id = self.after(delay, self._pollMouse)

        if self._shiftLeftMousePressed:
            self.add(10)
            self.after_id = self.after(delay, self._pollMouse)

        if self._rightMousePressed:
            self.sub(1)
            self.after_id = self.after(delay, self._pollMouse)

        if self._shiftRightMousePressed:
            self.sub(10)
            self.after_id = self.after(delay, self._pollMouse)