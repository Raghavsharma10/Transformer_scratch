def get_airport_details(self, iata, page=1, limit=100):
        """Retrieve the details of an airport

        Given the IATA code of an airport, this method returns the detailed information like lat lon, full name, URL, codes etc.

        Args:
            iata (str): The IATA code for an airport, e.g. HYD
            page (int): Optional page number; for users who are on a plan with flightradar24 they can pass in higher page numbers to get more data
            limit (int): Optional limit on number of records returned

        Returns:
            A list of dicts with the data; one dict for each row of data from flightradar24

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_airport_details('HYD')
            f.get_airport_details('HYD',page=1,limit=10)

        """
        url = AIRPORT_DATA_BASE.format(iata, str(self.AUTH_TOKEN), page, limit)
        details = self._fr24.get_airport_details(url)
        weather = self._fr24.get_airport_weather(url)
        # weather has more correct and standard elevation details in feet and meters
        details['position']['elevation'] = weather['elevation']
        return details