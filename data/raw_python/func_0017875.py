def get_fleet(self, airline_key):
        """Get the fleet for a particular airline.

        Given a airline code form the get_airlines() method output, this method returns the fleet for the airline.

        Args:
            airline_key (str): The code for the airline on flightradar24

        Returns:
            A list of dicts, one for each aircraft in the airlines fleet

        Example::
            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_fleet('ai-aic')
        """
        url = AIRLINE_FLEET_BASE.format(airline_key)
        return self._fr24.get_airline_fleet_data(url, self.AUTH_TOKEN != '')