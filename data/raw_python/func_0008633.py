def new(self, size, fill):
        """Return a new Image instance filled with a color."""
        return Image(PIL.Image.new("RGB", size, fill))