def get_flights(self, search_key):
        """Get the flights for a particular airline.

        Given a full or partial flight number string, this method returns the first 100 flights matching that string.

        Please note this method was different in earlier versions. The older versions took an airline code and returned all scheduled flights for that airline

        Args:
            search_key (str): Full or partial flight number for any airline e.g. MI47 to get all SilkAir flights starting with MI47

        Returns:
            A list of dicts, one for each scheduled flight in the airlines network

        Example::
            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_flights('MI47')
        """
        # assume limit 100 to return first 100 of any wild card search
        url = AIRLINE_FLT_BASE.format(search_key, 100)
        return self._fr24.get_airline_flight_data(url)