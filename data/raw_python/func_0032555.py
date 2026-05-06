def get_calendar(self, **kwargs):
        """Obtain EMT calendar for a range of dates.

        Args:
            start_day (int): Starting day of the month in format DD.
                The number is automatically padded if it only has one digit.
            start_month (int): Starting month number in format MM.
                The number is automatically padded if it only has one digit.
            start_year (int): Starting year number in format YYYY.
            end_day (int): Ending day of the month in format DD.
                The number is automatically padded if it only has one digit.
            end_month (int): Ending month number in format MM.
                The number is automatically padded if it only has one digit.
            end_year (int): Ending year number in format YYYY.

        Returns:
            Status boolean and parsed response (list[CalendarItem]), or message
            string in case of error.
        """
        # Endpoint parameters
        start_date = util.date_string(
            kwargs.get('start_day', '01'),
            kwargs.get('start_month', '01'),
            kwargs.get('start_year', '1970')
        )

        end_date = util.date_string(
            kwargs.get('end_day', '01'),
            kwargs.get('end_month', '01'),
            kwargs.get('end_year', '1970')
        )

        params = {'SelectDateBegin': start_date, 'SelectDateEnd': end_date}

        # Request
        result = self.make_request('bus', 'get_calendar', **params)

        if not util.check_result(result):
            return False, result.get('resultDescription', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'resultValues')
        return True, [emtype.CalendarItem(**a) for a in values]