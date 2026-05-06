def get_airport_metars_hist(self, iata):
        """Retrieve the metar data for past 72 hours. The data will not be parsed to readable format.

        Given the IATA code of an airport, this method returns the metar information for last 72 hours.

        Args:
            iata (str): The IATA code for an airport, e.g. HYD

        Returns:
            The metar data for the airport

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_airport_metars_hist('HYD')

        """
        url = AIRPORT_BASE.format(iata) + "/weather"
        return self._fr24.get_airport_metars_hist(url)