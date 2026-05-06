def _render(self):
        """
        Render the text.

        Avoid using this fonction too many time as it is slow as it is low to render text and blit it.
        """

        self._last_text = self.text

        self._surface = self.font.render(self.text, True, self.color, self.bg_color)
        rect = self._surface.get_rect()

        self.size = rect.size