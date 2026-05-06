def get_ride_types(self, latitude, longitude, ride_type=None):
        """Get information about the Ride Types offered by Lyft at a given location.
        Parameters
            latitude (float)
                The latitude component of a location.
            longitude (float)
                The longitude component of a location.
            ride_type (str)
                Optional specific ride type information only.
        Returns
            (Response)
                A Response object containing available ride_type(s) information.
        """
        args = OrderedDict([
            ('lat', latitude),
            ('lng', longitude),
            ('ride_type', ride_type),
        ])

        return self._api_call('GET', 'v1/ridetypes', args=args)