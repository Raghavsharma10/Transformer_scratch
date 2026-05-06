def remove(self):
        """
        Remove the layer artist for good
        """
        self._multivol.deallocate(self.id)
        ARRAY_CACHE.pop(self.id, None)
        PIXEL_CACHE.pop(self.id, None)