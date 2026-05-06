def get_pickup_time_estimates(self, latitude, longitude, ride_type=None):
        """Get pickup time estimates (ETA) for products at a given location.
        Parameters
            latitude (float)
                The latitude component of a location.
            longitude (float)
                The longitude component of a location.
            ride_type (str)
                Optional specific ride type pickup estimate only.
        Returns
            (Response)
                A Response containing each product's pickup time estimates.
        """
        args = OrderedDict([
            ('lat', latitude),
            ('lng', longitude),
            ('ride_type', ride_type),
        ])

        return self._api_call('GET', 'v1/eta', args=args)