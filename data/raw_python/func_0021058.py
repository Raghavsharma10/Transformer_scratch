def places_within_radius(
        self, place=None, latitude=None, longitude=None, radius=0, **kwargs
    ):
        """
        Return descriptions of the places stored in the collection that are
        within the circle specified by the given location and radius.
        A list of dicts will be returned.

        The center of the circle can be specified by the identifier of another
        place in the collection with the *place* keyword argument.
        Or, it can be specified by using both the *latitude* and *longitude*
        keyword arguments.

        By default the *radius* is given in kilometers, but you may also set
        the *unit* keyword argument to ``'m'``, ``'mi'``, or ``'ft'``.

        Limit the number of results returned with the *count* keyword argument.

        Change the sorted order by setting the *sort* keyword argument to
        ``b'DESC'``.
        """
        kwargs['withdist'] = True
        kwargs['withcoord'] = True
        kwargs['withhash'] = False
        kwargs.setdefault('sort', 'ASC')
        unit = kwargs.setdefault('unit', 'km')

        # Make the query
        if place is not None:
            response = self.redis.georadiusbymember(
                self.key, self._pickle(place), radius, **kwargs
            )
        elif (latitude is not None) and (longitude is not None):
            response = self.redis.georadius(
                self.key, longitude, latitude, radius, **kwargs
            )
        else:
            raise ValueError(
                'Must specify place, or both latitude and longitude'
            )

        # Assemble the result
        ret = []
        for item in response:
            ret.append(
                {
                    'place': self._unpickle(item[0]),
                    'distance': item[1],
                    'unit': unit,
                    'latitude': item[2][1],
                    'longitude': item[2][0],
                }
            )

        return ret