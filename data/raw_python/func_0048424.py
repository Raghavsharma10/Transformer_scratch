def default(self, value):
        """Set the default parameter and regenerate the thumbnail link."""
        self._default = value
        self._thumb  = self._link_to_img()