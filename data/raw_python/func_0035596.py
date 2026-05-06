def get_cost_estimates(
        self,
        start_latitude,
        start_longitude,
        end_latitude=None,
        end_longitude=None,
        ride_type=None,
    ):
        """Get cost estimates (in cents) for rides at a given location.
        Parameters
            start_latitude (float)
                The latitude component of a start location.
            start_longitude (float)
                The longitude component of a start location.
            end_latitude (float)
                Optional latitude component of a end location.
                If the destination parameters are not supplied, the endpoint will
                simply return the Prime Time pricing at the specified location.
            end_longitude (float)
                Optional longitude component of a end location.
                If the destination parameters are not supplied, the endpoint will
                simply return the Prime Time pricing at the specified location.
             ride_type (str)
                Optional specific ride type price estimate only.
        Returns
            (Response)
                A Response object containing each product's price estimates.
        """
        args = OrderedDict([
            ('start_lat', start_latitude),
            ('start_lng', start_longitude),
            ('end_lat', end_latitude),
            ('end_lng', end_longitude),
            ('ride_type', ride_type),
        ])

        return self._api_call('GET', 'v1/cost', args=args)