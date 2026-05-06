def get_stops_line(self, **kwargs):
        """Obtain information on the stops of the given lines.

        Arguments:
            lines (list[int] | int): Lines to query, may be empty to get
                all the lines.
            direction (str): Optional, either *forward* or *backward*.
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[Stop]), or message string
            in case of error.
        """
        # Endpoint parameters
        params = {
            'line': util.ints_to_string(kwargs.get('lines', [])),
            'direction': util.direction_code(kwargs.get('direction', '')),
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_stops_line', **params)

        # Funny endpoint, no status code
        # Only interested in 'stop'
        if not util.check_result(result, 'stop'):
            return False, 'UNKNOWN ERROR'

        # Parse
        values = util.response_list(result, 'stop')
        return True, [emtype.Stop(**a) for a in values]