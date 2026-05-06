def get_poi(self, **kwargs):
        """Obtain a list of POI in the given radius.

        Args:
            latitude (double): Latitude in decimal degrees.
            longitude (double): Longitude in decimal degrees.
            types (list[int] | int): POI IDs (or empty list to get all).
            radius (int): Radius (in meters) of the search.
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[Poi]), or message string
            in case of error.
        """
        # Endpoint parameters
        params = {
            'coordinateX': kwargs.get('longitude'),
            'coordinateY': kwargs.get('latitude'),
            'tipos': util.ints_to_string(kwargs.get('types')),
            'Radius': kwargs.get('radius'),
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_poi', **params)

        # Funny endpoint, no status code
        if not util.check_result(result, 'poiList'):
            return False, 'UNKNOWN ERROR'

        # Parse
        values = util.response_list(result, 'poiList')
        return True, [emtype.Poi(**a) for a in values]