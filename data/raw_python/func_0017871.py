def get_history_by_flight_number(self, flight_number, page=1, limit=100):
        """Fetch the history of a flight by its number.

        This method can be used to get the history of a flight route by the number.
        It checks the user authentication and returns the data accordingly.

        Args:
            flight_number (str): The flight number, e.g. AI101
            page (int): Optional page number; for users who are on a plan with flightradar24 they can pass in higher page numbers to get more data
            limit (int): Optional limit on number of records returned

        Returns:
            A list of dicts with the data; one dict for each row of data from flightradar24

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_history_by_flight_number('AI101')
            f.get_history_by_flight_number('AI101',page=1,limit=10)

        """
        url = FLT_BASE.format(flight_number, str(self.AUTH_TOKEN), page, limit)
        return self._fr24.get_data(url)