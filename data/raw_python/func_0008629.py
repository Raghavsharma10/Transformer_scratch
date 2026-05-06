def size(self):
        """``(width, height)`` in pixels."""
        if self._size and not self._pil_image:
            return self._size
        else:
            return self.pil_image.size