def roll(self):
        """ Increase the age of all items in the cache by 1. Items whose age
        is greater than self.max_age will be removed from the cache.
        """
        rem = []
        for key, item in self._cache.items():
            if item[0] > self.max_age:
                rem.append(key)
            item[0] += 1

        for key in rem:
            logger.debug("TransformCache remove: %s", key)
            del self._cache[key]