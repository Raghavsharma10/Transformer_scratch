def render(self, display):
        """Render basicly the text."""
        # to handle changing objects / callable
        if self.text != self._last_text:
            self._render()

        display.blit(self._surface, (self.topleft, self.size))