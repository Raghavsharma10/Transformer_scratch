def _draw(self):
        """
        Call the drawing API for the main menu widget with the current known
        terminal size and the terminal.
        """
        self._window.draw(self._width, self._height, self.terminal)