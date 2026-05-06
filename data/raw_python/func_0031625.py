def datetime_to_timestamp(self, dt):
        """Helper function to convert a datetime object to a timestamp.

        If datetime instance ``dt`` is naive, it is assumed that it is in UTC.

        In Python 3, this just calls ``datetime.timestamp()``, in Python 2, it substracts any timezone offset
        and returns the difference since 1970-01-01 00:00:00.

        Note that the function always returns an int, even in Python 3.

        >>> XmppBackendBase().datetime_to_timestamp(datetime(2017, 9, 17, 19, 59))
        1505678340
        >>> XmppBackendBase().datetime_to_timestamp(datetime(1984, 11, 6, 13, 21))
        468595260

        :param dt: The datetime object to convert. If ``None``, returns the current time.
        :type  dt: datetime
        :return: The seconds in UTC.
        :rtype: int
        """
        if dt is None:
            return int(time.time())

        if six.PY3:
            if not dt.tzinfo:
                dt = pytz.utc.localize(dt)
            return int(dt.timestamp())
        else:
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None) - dt.utcoffset()
            return int((dt - datetime(1970, 1, 1)).total_seconds())