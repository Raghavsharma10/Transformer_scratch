def check_type(param, datatype):
    """
    Make sure that param is of type datatype and return it.
    If param is None, return it.
    If param is an instance of datatype, return it.
    If param is not an instance of datatype and is not None, cast it as
    datatype and return it.
    """
    if param is None:
        return param

    if getattr(datatype, 'clean', None) and callable(datatype.clean):
        try:
            return datatype.clean(param)
        except ValueError:
            raise BadArgumentError(param)

    elif isinstance(datatype, str):
        # You've given it something like `'bool'` as a string.
        # This is the legacy way of doing it.
        datatype = {
            'str': str,
            'bool': bool,
            'float': float,
            'date': datetime.date,
            'datetime': datetime.datetime,
            'timedelta': datetime.timedelta,
            'json': 'json',  # exception
            'int': int,
        }[datatype]

    if datatype is str and not isinstance(param, basestring):
        try:
            param = str(param)
        except ValueError:
            param = str()

    elif datatype is int and not isinstance(param, int):
        try:
            param = int(param)
        except ValueError:
            param = int()

    elif datatype is bool and not isinstance(param, bool):
        param = str(param).lower() in ("true", "t", "1", "y", "yes")

    elif (
        datatype is datetime.datetime and
        not isinstance(param, datetime.datetime)
    ):
        try:
            param = dtutil.string_to_datetime(param)
        except ValueError:
            param = None

    elif datatype is datetime.date and not isinstance(param, datetime.date):
        try:
            param = dtutil.string_to_datetime(param).date()
        except ValueError:
            param = None

    elif (
        datatype is datetime.timedelta and
        not isinstance(param, datetime.timedelta)
    ):
        try:
            param = dtutil.strHoursToTimeDelta(param)
        except ValueError:
            param = None

    elif datatype == "json" and isinstance(param, basestring):
        try:
            param = json.loads(param)
        except ValueError:
            param = None

    return param