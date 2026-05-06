def get_groups(self, **kwargs):
        """Obtain line types and details.

        Args:
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[GeoGroupItem]), or message
            string in case of error.
        """
        # Endpoint parameters
        params = {
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_groups', **params)

        if not util.check_result(result):
            return False, result.get('resultDescription', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'resultValues')
        return True, [emtype.GeoGroupItem(**a) for a in values]