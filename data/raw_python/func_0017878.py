def get_airport_weather(self, iata, page=1, limit=100):
        """Retrieve the weather at an airport

        Given the IATA code of an airport, this method returns the weather information.

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
            f.get_airport_weather('HYD')
            f.get_airport_weather('HYD',page=1,limit=10)

        """
        url = AIRPORT_DATA_BASE.format(iata, str(self.AUTH_TOKEN), page, limit)
        weather = self._fr24.get_airport_weather(url)
        mi = weather['sky']['visibility']['mi']
        if (mi is not None) and (mi != "None"):
            mi = float(mi)
            km = mi * 1.6094
            weather['sky']['visibility']['km'] = km
        return weather