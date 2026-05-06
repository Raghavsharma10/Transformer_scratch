def cursor_pos(self):
        """The cursor position in pixels."""
        if len(self) == 0:
            return self.left + self.default_text.get_width()

        papy = self._surface.get_width()
        if papy > self.w:
            shift = papy - self.width
        else:
            shift = 0

        return self.left + self.font.size(self.shawn_text[:self.cursor])[0] - shift