def set_status(self, instance, status):
        """Sets the field status for up to 5 minutes."""
        status_key = self.get_status_key(instance)
        cache.set(status_key, status, timeout=300)