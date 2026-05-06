def sync(self, clear_cache=False):
        """
        Copy items from the local cache to the persistent Dict.
        If *clear_cache* is ``True``, clear out the local cache after
        pushing its items to Redis.
        """
        self.persistence.update(self)

        if clear_cache:
            self.cache.clear()