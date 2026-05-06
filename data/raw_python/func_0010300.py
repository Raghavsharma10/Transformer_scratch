def to_none_or_dt(input):
    """Convert ``input`` to either None or a datetime object

    If the input is None, None will be returned.
    If the input is a datetime object, it will be converted to a datetime
    object with UTC timezone info.  If the datetime object is naive, then
    this method will assume the object is specified according to UTC and
    not local or some other timezone.
    If the input to the function is a string, this method will attempt to
    parse the input as an ISO-8601 formatted string.

    :param input: Input data (expected to be either str, None, or datetime object)
    :return: datetime object from input or None if already None
    :rtype: datetime or None

    """
    if input is None:
        return input
    elif isinstance(input, datetime.datetime):
        arrow_dt = arrow.Arrow.fromdatetime(input, input.tzinfo or 'utc')
        return arrow_dt.to('utc').datetime
    if isinstance(input, six.string_types):
        # try to convert from ISO8601
        return iso8601_to_dt(input)
    else:
        raise TypeError("Not a string, NoneType, or datetime object")