def distance_between(self, place_1, place_2, unit='km'):
        """
        Return the great-circle distance between *place_1* and *place_2*,
        in the *unit* specified.

        The default unit is ``'km'``, but ``'m'``, ``'mi'``, and ``'ft'`` can
        also be specified.
        """
        pickled_place_1 = self._pickle(place_1)
        pickled_place_2 = self._pickle(place_2)
        try:
            return self.redis.geodist(
                self.key, pickled_place_1, pickled_place_2, unit=unit
            )
        except TypeError:
            return None