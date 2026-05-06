def cancel_ride(self, ride_id, cancel_confirmation_token=None):
        """Cancel an ongoing ride on behalf of a user.
        Params
            ride_id (str)
                The unique ID of the Ride Request.
            cancel_confirmation_token (str)
                Optional string containing the cancellation confirmation token.
        Returns
            (Response)
                A Response object with successful status_code
                if ride was canceled.
        """
        args = {
            "cancel_confirmation_token": cancel_confirmation_token
        }
        endpoint = 'v1/rides/{}/cancel'.format(ride_id)
        return self._api_call('POST', endpoint, args=args)