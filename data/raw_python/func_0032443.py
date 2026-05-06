def list_types_poi(self, **kwargs):
        """Obtain a list of families, types and categories of POI.

        Args:
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[ParkingPoiType]), or message
            string in case of error.
        """
        # Endpoint parameters
        url_args = {'language': util.language_code(kwargs.get('lang'))}

        # Request
        result = self.make_request('list_poi_types', url_args)

        if not util.check_result(result):
            return False, result.get('message', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'Data')
        return True, [emtype.ParkingPoiType(**a) for a in values]