def icon_description(self, **kwargs):
        """Obtain a list of elements that have an associated icon.

        Args:
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[IconDescription]), or
            message string in case of error.
        """
        # Endpoint parameters
        params = {'language': util.language_code(kwargs.get('lang'))}

        # Request
        result = self.make_request('icon_description', {}, **params)

        if not util.check_result(result):
            return False, result.get('message', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'Data')
        return True, [emtype.IconDescription(**a) for a in values]