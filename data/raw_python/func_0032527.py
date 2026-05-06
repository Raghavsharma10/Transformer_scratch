def get_poi_types(self, **kwargs):
        """Obtain POI types.

        Args:
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[PoiType]), or message string
            in case of error.
        """
        # Endpoint parameters
        params = {
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_poi_types', **params)

        # Parse
        values = result.get('types', [])
        return True, [emtype.PoiType(**a) for a in values]