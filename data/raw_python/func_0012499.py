def datetime_from_iso(iso_string):
    """
    Create a DateTime object from a ISO string

    .. code :: python

        reusables.datetime_from_iso('2017-03-10T12:56:55.031863')
        datetime.datetime(2017, 3, 10, 12, 56, 55, 31863)

    :param iso_string: string of an ISO datetime
    :return: DateTime object
    """
    try:
        assert datetime_regex.datetime.datetime.match(iso_string).groups()[0]
    except (ValueError, AssertionError, IndexError, AttributeError):
        raise TypeError("String is not in ISO format")
    try:
        return datetime.datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S")