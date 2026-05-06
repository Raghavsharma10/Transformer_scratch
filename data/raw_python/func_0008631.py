def convert(self, *formats):
        """Return an Image instance with the first matching format.

        For each format in ``*args``: If the image's :attr:`format` attribute
        is the same as the format, return self, otherwise try the next format.

        If none of the formats match, return a new Image instance with the
        last format.

        """
        for format in formats:
            format = Image.image_format(format)
            if self.format == format:
                return self
        else:
            return self._convert(format)