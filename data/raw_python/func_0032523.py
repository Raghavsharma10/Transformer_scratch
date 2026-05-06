def get_arrive_stop(self, **kwargs):
        """Obtain bus arrival info in target stop.

        Args:
            stop_number (int): Stop number to query.
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[Arrival]), or message string
            in case of error.
        """
        # Endpoint parameters
        params = {
            'idStop': kwargs.get('stop_number'),
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_arrive_stop', **params)

        # Funny endpoint, no status code
        if not util.check_result(result, 'arrives'):
            return False, 'UNKNOWN ERROR'

        # Parse
        values = util.response_list(result, 'arrives')
        return True, [emtype.Arrival(**a) for a in values]