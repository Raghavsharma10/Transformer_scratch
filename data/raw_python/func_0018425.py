def set_font_size(self, pt=None, px=None):
        """Set the font size to the desired size, in pt or px."""
        self.font.set_size(pt, px)

        self._render()