def _date2int(date):
        """
        Returns an integer representation of a date.

        :param str|datetime.date date: The date.

        :rtype: int
        """
        if isinstance(date, str):
            if date.endswith(' 00:00:00') or date.endswith('T00:00:00'):
                # Ignore time suffix.
                date = date[0:-9]
            tmp = datetime.datetime.strptime(date, '%Y-%m-%d')
            return tmp.toordinal()

        if isinstance(date, datetime.date):
            return date.toordinal()

        if isinstance(date, int):
            return date

        raise ValueError('Unexpected type {0!s}'.format(date.__class__))