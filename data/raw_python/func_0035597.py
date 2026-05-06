def get_drivers(self, latitude, longitude):
        """Get information about the location of drivers available near a location.
        A list of 5 locations for a sample of drivers for each ride type will be provided.
        Parameters
            latitude (float)
                The latitude component of a location.
            longitude (float)
                The longitude component of a location.
        Returns
            (Response)
                A Response object containing available drivers information
                near the specified location.
        """
        args = OrderedDict([
            ('lat', latitude),
            ('lng', longitude),
        ])

        return self._api_call('GET', 'v1/drivers', args=args)