def set(self, hue):
        """Set cursor position on the color corresponding to the hue value."""
        x = hue / 360. * self.winfo_width()
        self.coords('cursor', x, 0, x, self.winfo_height())
        self._variable.set(hue)