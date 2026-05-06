def get_street(self, **kwargs):
        """Obtain a list of nodes related to a location within a given radius.

        Not sure of its use, but...

        Args:
            street_name (str): Name of the street to search.
            street_number (int): Street number to search.
            radius (int): Radius (in meters) of the search.
            stops (int): Number of the stop to search.
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[Site]), or message string
            in case of error.
        """
        # Endpoint parameters
        params = {
            'description': kwargs.get('street_name'),
            'streetNumber': kwargs.get('street_number'),
            'Radius': kwargs.get('radius'),
            'Stops': kwargs.get('stops'),
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_street', **params)

        # Funny endpoint, no status code
        if not util.check_result(result, 'site'):
            return False, 'UNKNOWN ERROR'

        # Parse
        values = util.response_list(result, 'site')
        return True, [emtype.Site(**a) for a in values]