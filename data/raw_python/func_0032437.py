def detail_poi(self, **kwargs):
        """Obtain detailed info of a given POI.

        Args:
            family (str): Family code of the POI (3 chars).
            lang (str): Language code (*es* or *en*).
            id (int): Optional, ID of the POI to query. Passing value -1 will
                result in information from all POIs.

        Returns:
            Status boolean and parsed response (list[PoiDetails]), or
            message string in case of error.
        """
        # Endpoint parameters
        params = {
            'language': util.language_code(kwargs.get('lang')),
            'family': kwargs.get('family')
        }

        if kwargs.get('id'):
            params['id'] = kwargs['id']

        # Request
        result = self.make_request('detail_poi', {}, **params)

        if not util.check_result(result):
            return False, result.get('message', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'Data')
        return True, [emtype.PoiDetails(**a) for a in values]