def secure(self, value):
        """Set the secure parameter and regenerate the thumbnail link."""
        self._secure = value
        self._thumb  = self._link_to_img()