def get_info_line(self, **kwargs):
        """Obtain basic information on a bus line on a given date.

        Args:
            day (int): Day of the month in format DD.
                The number is automatically padded if it only has one digit.
            month (int): Month number in format MM.
                The number is automatically padded if it only has one digit.
            year (int): Year number in format YYYY.
            lines (list[int] | int): Lines to query, may be empty to get
                all the lines.
            lang (str): Language code (*es* or *en*).

        Returns:
            Status boolean and parsed response (list[Line]), or message string
            in case of error.
        """
        # Endpoint parameters
        select_date = '%02d/%02d/%d' % (
            kwargs.get('day', '01'),
            kwargs.get('month', '01'),
            kwargs.get('year', '1970')
        )

        params = {
            'fecha': select_date,
            'line': util.ints_to_string(kwargs.get('lines', [])),
            'cultureInfo': util.language_code(kwargs.get('lang'))
        }

        # Request
        result = self.make_request('geo', 'get_info_line', **params)

        # Funny endpoint, no status code
        if not util.check_result(result, 'Line'):
            return False, 'UNKNOWN ERROR'

        # Parse
        values = util.response_list(result, 'Line')
        return True, [emtype.Line(**a) for a in values]