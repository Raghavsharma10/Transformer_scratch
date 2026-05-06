def _render(self):
        """
        Render the text.

        Avoid using this fonction too many times as it is slow as it is slow to render text and blit it.
        """

        self._last_text = self.text

        self._surface = self.font.render(self.text, True, self.color, self.bg_color)
        size = self.width, self._surface.get_height()
        self.size = size