def list_parking(self, **kwargs):
        """Obtain a list of parkings.

        Args:
            lang (str):  Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[Parking]), or message
            string in case of error.
        """
        # Endpoint parameters
        url_args = {'lang': util.language_code(kwargs.get('lang'))}

        # Request
        result = self.make_request('list_parking', url_args)

        if not util.check_result(result):
            return False, result.get('message', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'Data')
        return True, [emtype.Parking(**a) for a in values]