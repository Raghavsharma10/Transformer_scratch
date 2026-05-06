def http_date(value):
    """ Formats the @value in required HTTP style

        @value: :class:datetime.datetime, #int, #float, #str time-like object

        -> #str HTTP-style formatted date

        (c)2014, Marcel Hellkamp
    """
    if isinstance(value, datetime.datetime):
        value = value.utctimetuple()
    elif isinstance(value, (int, float)):
        value = time.gmtime(value)
    if not isinstance(value, str):
        value = time.strftime("%a, %d %b %Y %H:%M:%S GMT", value)
    return value