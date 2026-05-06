def color(self, value):
        """Set the color to a new value (tuple). Renders the text if needed."""

        if value != self.color:
            self._color = value
            self._render()