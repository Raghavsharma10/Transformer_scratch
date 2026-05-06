def get_hash(self, place):
        """
        Return the Geohash of *place*.
        If it's not present in the collection, ``None`` will be returned
        instead.
        """
        pickled_place = self._pickle(place)
        try:
            return self.redis.geohash(self.key, pickled_place)[0]
        except (AttributeError, TypeError):
            return None