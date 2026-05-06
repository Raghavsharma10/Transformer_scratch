def format(self):
        """The format of the image file.

        An uppercase string corresponding to the
        :attr:`PIL.ImageFile.ImageFile.format` attribute.  Valid values include
        ``"JPEG"`` and ``"PNG"``.

        """
        if self._format:
            return self._format
        elif self.pil_image:
            return self.pil_image.format