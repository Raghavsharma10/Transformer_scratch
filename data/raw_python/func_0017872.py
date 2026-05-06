def get_history_by_tail_number(self, tail_number, page=1, limit=100):
        """Fetch the history of a particular aircraft by its tail number.

        This method can be used to get the history of a particular aircraft by its tail number.
        It checks the user authentication and returns the data accordingly.

        Args:
            tail_number (str): The tail number, e.g. VT-ANL
            page (int): Optional page number; for users who are on a plan with flightradar24 they can pass in higher page numbers to get more data
            limit (int): Optional limit on number of records returned

        Returns:
            A list of dicts with the data; one dict for each row of data from flightradar24

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_history_by_flight_number('VT-ANL')
            f.get_history_by_flight_number('VT-ANL',page=1,limit=10)

        """
        url = REG_BASE.format(tail_number, str(self.AUTH_TOKEN), page, limit)
        return self._fr24.get_data(url, True)