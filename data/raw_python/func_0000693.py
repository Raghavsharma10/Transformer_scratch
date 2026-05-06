def refresh(self, *args, **kwargs):
        """
        Fetch the result SYNCHRONOUSLY and populate the cache
        """
        result = self.fetch(*args, **kwargs)
        self.store(self.key(*args, **kwargs), self.expiry(*args, **kwargs), result)
        return result