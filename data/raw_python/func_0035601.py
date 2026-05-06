def get_user_ride_history(self, start_time, end_time, limit=None):
        """Get activity about the user's lifetime activity with Lyft.
        Parameters
            start_time (datetime)
                Restrict to rides starting after this point in time.
                The earliest supported date is 2015-01-01T00:00:00Z
            end_time (datetime)
                Optional Restrict to rides starting before this point in time.
                The earliest supported date is 2015-01-01T00:00:00Z
            limit (int)
                Optional integer amount of results to return. Default is 10.
        Returns
            (Response)
                A Response object containing ride history.
        """
        args = {
            'start_time': start_time,
            'end_time': end_time,
            'limit': limit,
        }

        return self._api_call('GET', 'v1/rides', args=args)