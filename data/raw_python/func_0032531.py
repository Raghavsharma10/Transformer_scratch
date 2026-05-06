def get_street_from_xy(self, **kwargs):
        """Obtain a list of streets around the specified point.

        Args:
            latitude (double): Latitude in decimal degrees.
            longitude (double): Longitude in decimal degrees.
            radius (int): Radius (in meters) of the search.
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[Street]), or message string
            in case of error.
        """
        # Endpoint parameters
        params = {
            'coordinateX': kwargs.get('longitude'),
            'coordinateY': kwargs.get('latitude'),
            'Radius': kwargs.get('radius'),
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_street_from_xy', **params)

        # Funny endpoint, no status code
        if not util.check_result(result, 'site'):
            return False, 'UNKNOWN ERROR'

        # Parse
        values = util.response_list(result, 'site')
        return True, [emtype.Street(**a) for a in values]