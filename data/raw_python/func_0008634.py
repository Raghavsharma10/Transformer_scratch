def resize(self, size):
        """Return a new Image instance with the given size."""
        return Image(self.pil_image.resize(size, PIL.Image.ANTIALIAS))