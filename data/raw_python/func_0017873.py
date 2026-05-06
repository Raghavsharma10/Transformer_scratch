def get_airports(self, country):
        """Returns a list of all the airports
        For a given country this returns a list of dicts, one for each airport, with information like the iata code of the airport etc

        Args:
            country (str): The country for which the airports will be fetched

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            f.get_airports('India')

        """
        url = AIRPORT_BASE.format(country.replace(" ", "-"))
        return self._fr24.get_airports_data(url)