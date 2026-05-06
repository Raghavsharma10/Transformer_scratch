def _get_date_type(date):
        """
        Returns the type of a date.

        :param str|datetime.date date: The date.

        :rtype: str
        """
        if isinstance(date, str):
            return 'str'

        if isinstance(date, datetime.date):
            return 'date'

        if isinstance(date, int):
            return 'int'

        raise ValueError('Unexpected type {0!s}'.format(date.__class__))