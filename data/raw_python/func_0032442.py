def list_street_poi_parking(self, **kwargs):
        """Obtain a list of addresses and POIs.

        This endpoint uses an address to perform the search

        Args:
            lang (str): Language code (*es* or *en*).
            address (str): Address in which to perform the search.

        Returns:
            Status boolean and parsed response (list[ParkingPoi]), or message
            string in case of error.
        """
        # Endpoint parameters
        url_args = {
            'language': util.language_code(kwargs.get('lang')),
            'address': kwargs.get('address', '')
        }

        # Request
        result = self.make_request('list_street_poi_parking', url_args)

        if not util.check_result(result):
            return False, result.get('message', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'Data')
        return True, [emtype.ParkingPoi(**a) for a in values]