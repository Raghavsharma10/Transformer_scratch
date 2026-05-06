def set_location(self, location):
        """Set the location for this data point

        The location must be either None (if no location data is known) or a
        3-tuple of floating point values in the form
        (latitude-degrees, longitude-degrees, altitude-meters).

        """
        if location is None:
            self._location = location

        elif isinstance(location, *six.string_types):  # from device cloud, convert from csv
            parts = str(location).split(",")
            if len(parts) == 3:
                self._location = tuple(map(float, parts))
                return
            else:
                raise ValueError("Location string %r has unexpected format" % location)

        # TODO: could maybe try to allow any iterable but this covers the most common cases
        elif (isinstance(location, (tuple, list))
                and len(location) == 3
                and all([isinstance(x, (float, six.integer_types)) for x in location])):
            self._location = tuple(map(float, location))  # coerce ints to float
        else:
            raise TypeError("Location must be None or 3-tuple of floats")

        self._location = location