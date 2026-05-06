def detail_parking(self, **kwargs):
        """Obtain detailed info of a given parking.

        Args:
            lang (str):  Language code (*es* or *en*).
            day (int): Day of the month in format DD.
                The number is automatically padded if it only has one digit.
            month (int): Month number in format MM.
                The number is automatically padded if it only has one digit.
            year (int): Year number in format YYYY.
            hour (int): Hour of the day in format hh.
                The number is automatically padded if it only has one digit.
            minute (int): Minute of the hour in format mm.
                The number is automatically padded if it only has one digit.
            parking (int): ID of the parking to query.
            family (str): Family code of the parking (3 chars).

        Returns:
            Status boolean and parsed response (list[ParkingDetails]), or message
            string in case of error.
        """
        # Endpoint parameters
        date = util.datetime_string(
            kwargs.get('day', 1),
            kwargs.get('month', 1),
            kwargs.get('year', 1970),
            kwargs.get('hour', 0),
            kwargs.get('minute', 0)
        )

        params = {
            'language': util.language_code(kwargs.get('lang')),
            'publicData': True,
            'date': date,
            'id': kwargs.get('parking'),
            'family': kwargs.get('family')
        }

        # Request
        result = self.make_request('detail_parking', {}, **params)

        if not util.check_result(result):
            return False, result.get('message', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'Data')
        return True, [emtype.ParkingDetails(**a) for a in values]