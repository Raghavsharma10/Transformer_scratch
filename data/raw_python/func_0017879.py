def get_airport_metars(self, iata, page=1, limit=100):
        """Retrieve the metar data at the current time

        Given the IATA code of an airport, this method returns the metar information.

        Args:
            iata (str): The IATA code for an airport, e.g. HYD
            page (int): Optional page number; for users who are on a plan with flightradar24 they can pass in higher page numbers to get more data
            limit (int): Optional limit on number of records returned

        Returns:
            The metar data for the airport

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_airport_metars('HYD')

        """
        url = AIRPORT_DATA_BASE.format(iata, str(self.AUTH_TOKEN), page, limit)
        w = self._fr24.get_airport_weather(url)
        return w['metar']