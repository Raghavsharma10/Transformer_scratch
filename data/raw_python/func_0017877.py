def get_flights_from_to(self, origin, destination):
        """Get the flights for a particular origin and destination.

        Given an origin and destination this method returns the upcoming scheduled flights between these two points.
        The data returned has the airline, airport and schedule information - this is subject to change in future.

        Args:
            origin (str): The origin airport code
            destination (str): The destination airport code

        Returns:
            A list of dicts, one for each scheduled flight between the two points.

        Example::
            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_flights_from_to('SIN','HYD')
        """
        # assume limit 100 to return first 100 of any wild card search
        url = AIRLINE_FLT_BASE_POINTS.format(origin, destination)
        return self._fr24.get_airline_flight_data(url, by_airports=True)