def copy(self, key=None):
        """
        Return a new collection with the same items as this one.
        If *key* is specified, create the new collection with the given
        Redis key.
        """
        other = self.__class__(redis=self.redis, key=key)
        other.update(self)

        return other