def rating(self, value):
        """Set the rating parameter and regenerate the thumbnail link."""
        self._rating = value
        self._thumb  = self._link_to_img()