def _convert(self, format):
        """Return a new Image instance with the given format.

        Returns self if the format is already the same.

        """
        if self.format == format:
            return self
        else:
            image = Image(self.pil_image)
            image._format = format
            return image