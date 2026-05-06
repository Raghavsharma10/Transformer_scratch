def copy(self, key=None):
        """
        Creates another collection with the same items and maxsize with
        the given *key*.
        """
        other = self.__class__(
            maxsize=self.maxsize, redis=self.persistence.redis, key=key
        )
        other.update(self)

        return other