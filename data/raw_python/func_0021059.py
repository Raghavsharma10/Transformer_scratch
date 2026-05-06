def set_location(self, place, latitude, longitude, pipe=None):
        """
        Set the location of *place* to the location specified by
        *latitude* and *longitude*.

        *place* can be any pickle-able Python object.
        """
        pipe = self.redis if pipe is None else pipe
        pipe.geoadd(self.key, longitude, latitude, self._pickle(place))