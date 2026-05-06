def get_route_lines_route(self, **kwargs):
        """Obtain itinerary for one or more lines in the given date.

        Args:
            day (int): Day of the month in format DD.
                The number is automatically padded if it only has one digit.
            month (int): Month number in format MM.
                The number is automatically padded if it only has one digit.
            year (int): Year number in format YYYY.
            lines (list[int] | int): Lines to query, may be empty to get
                all the lines.

        Returns:
            Status boolean and parsed response (list[RouteLinesItem]), or message
            string in case of error.
        """
        # Endpoint parameters
        select_date = '%02d/%02d/%d' % (
            kwargs.get('day', '01'),
            kwargs.get('month', '01'),
            kwargs.get('year', '1970')
        )

        params = {
            'SelectDate': select_date,
            'Lines': util.ints_to_string(kwargs.get('lines', []))
        }

        # Request
        result = self.make_request('geo', 'get_route_lines_route', **params)

        if not util.check_result(result):
            return False, result.get('resultDescription', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'resultValues')
        return True, [emtype.RouteLinesItem(**a) for a in values]