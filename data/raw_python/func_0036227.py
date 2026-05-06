def register_all(self, callback, user_data=None):
        """Register a callback for all sensors."""
        self._callback = callback
        self._callback_data = user_data