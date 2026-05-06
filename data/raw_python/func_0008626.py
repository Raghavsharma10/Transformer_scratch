def pil_image(self):
        """A :class:`PIL.Image.Image` instance containing the image data."""
        if not self._pil_image:
            if self._format == "SVG":
                raise VectorImageError("can't rasterise vector images")
            self._pil_image = PIL.Image.open(StringIO(self.contents))
        return self._pil_image