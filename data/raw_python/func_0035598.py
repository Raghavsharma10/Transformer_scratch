def request_ride(
        self,
        ride_type=None,
        start_latitude=None,
        start_longitude=None,
        start_address=None,
        end_latitude=None,
        end_longitude=None,
        end_address=None,
        primetime_confirmation_token=None,
    ):
        """Request a ride on behalf of an Lyft user.
        Parameters
            ride_type (str)
                Name of the type of ride you're requesting.
                E.g., lyft, lyft_plus
            start_latitude (float)
                Latitude component of a start location.
            start_longitude (float)
                Longitude component of a start location.
            start_address (str)
                Optional pickup address.
            end_latitude (float)
                Optional latitude component of a end location.
                Destination would be NULL in this case.
            end_longitude (float)
                Optional longitude component of a end location.
                Destination would be NULL in this case.
            end_address (str)
                Optional destination address.
            primetime_confirmation_token (str)
                Optional string containing the Prime Time confirmation token
                to book rides having Prime Time Pricing.
        Returns
            (Response)
                A Response object containing the ride request ID and other
                details about the requested ride..
        """
        args = {
            'ride_type': ride_type,
            'origin': {
                'lat': start_latitude,
                'lng': start_longitude,
                'address': start_address,
            },
            'destination': {
                'lat': end_latitude,
                'lng': end_longitude,
                'address': end_address,
            },
            'primetime_confirmation_token': primetime_confirmation_token,
        }

        return self._api_call('POST', 'v1/rides', args=args)