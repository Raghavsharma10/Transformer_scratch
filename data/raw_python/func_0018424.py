def bg_color(self, value):
        """Sets the color to a new value (tuple). Renders the text if needed."""

        if value != self.bg_color:
            self._bg_color = value
            self._render()