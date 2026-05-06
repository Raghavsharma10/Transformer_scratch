def parse_datetime(value):
    """Returns a datetime object for a given argument

    This helps to convert strings, dates and datetimes to proper tz-enabled
    datetime objects."""

    if isinstance(value, (string_types, text_type, binary_type)):
        value = dateutil.parser.parse(value)
        value.replace(tzinfo=dateutil.tz.tzutc())
        return value
    elif isinstance(value, datetime.datetime):
        value.replace(tzinfo=dateutil.tz.tzutc())
        return value
    elif isinstance(value, datetime.date):
        value = datetime.datetime(value.year, value.month, value.day)
        value.replace(tzinfo=dateutil.tz.tzutc())
        return value
    else:
        raise ValueError('Value must be parsable to datetime object. Got `{}`'.format(type(value)))