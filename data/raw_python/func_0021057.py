def get_location(self, place):
        """
        Return a dict with the coordinates *place*. The dict's keys are
        ``'latitude'`` and ``'longitude'``.
        If it's not present in the collection, ``None`` will be returned
        instead.
        """
        pickled_place = self._pickle(place)
        try:
            longitude, latitude = self.redis.geopos(self.key, pickled_place)[0]
        except (AttributeError, TypeError):
            return None

        return {'latitude': latitude, 'longitude': longitude}