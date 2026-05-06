def format_date(format_string=None, datetime_obj=None):
    """
    Format a datetime object with Java SimpleDateFormat's-like string.

    If datetime_obj is not given - use current datetime.
    If format_string is not given - return number of millisecond since epoch.

    :param format_string:
    :param datetime_obj:
    :return:
    :rtype string
    """
    datetime_obj = datetime_obj or datetime.now()
    if format_string is None:
        seconds = int(datetime_obj.strftime("%s"))
        milliseconds = datetime_obj.microsecond // 1000
        return str(seconds * 1000 + milliseconds)
    else:
        formatter = SimpleDateFormat(format_string)
        return formatter.format_datetime(datetime_obj)