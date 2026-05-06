def size(self, value):
        """Set the size parameter and regenerate the thumbnail link."""
        self._size   = value
        self._thumb  = self._link_to_img()