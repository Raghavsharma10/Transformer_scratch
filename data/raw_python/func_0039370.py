def getdatetime(timedateformat='complete'):
    """
    Get the current date or time in a specific format.

    :type timedateformat: string
    :param timedateformat: The type of date to query for. Can be: day, month, year, hour, minute, second, millisecond, yearmonthday, daymonthyear, hourminutesecond, secondminutehour, complete, datetime or timedate.
    """

    timedateformat = timedateformat.lower()
    if timedateformat == 'day':
        return ((str(datetime.datetime.now())).split(' ')[0]).split('-')[2]
    elif timedateformat == 'month':
        return ((str(datetime.datetime.now())).split(' ')[0]).split('-')[1]
    elif timedateformat == 'year':
        return ((str(datetime.datetime.now())).split(' ')[0]).split('-')[0]
    elif timedateformat == 'hour':
        return (((str(datetime.datetime.now())).split(' ')[1]).split('.')[0]
               ).split(':')[0]
    elif timedateformat == 'minute':
        return (((str(datetime.datetime.now())).split(' ')[1]).split('.')[0]
               ).split(':')[1]
    elif timedateformat == 'second':
        return (((str(datetime.datetime.now())).split(' ')[1]).split('.')[0]
               ).split(':')[2]
    elif timedateformat == 'millisecond':
        return (str(datetime.datetime.now())).split('.')[1]
    elif timedateformat == 'yearmonthday':
        return (str(datetime.datetime.now())).split(' ')[0]
    elif timedateformat == 'daymonthyear':
        return ((str(datetime.datetime.now())).split(' ')[0]).split(
            '-')[2] + '-' + ((str(
                datetime.datetime.now())).split(' ')[0]).split('-')[1] + '-' + (
                    (str(datetime.datetime.now())).split(' ')[0]).split('-')[0]
    elif timedateformat == 'hourminutesecond':
        return ((str(datetime.datetime.now())).split(' ')[1]).split('.')[0]
    elif timedateformat == 'secondminutehour':
        return (((str(datetime.datetime.now())).split(' ')[1]).split('.')[0]
               ).split(':')[2] + ':' + (((str(datetime.datetime.now())).split(
                   ' ')[1]).split('.')[0]).split(':')[1] + ':' + (
                       ((str(datetime.datetime.now())).split(' ')[1]
                       ).split('.')[0]).split(':')[0]
    elif timedateformat == 'complete':
        return str(datetime.datetime.now())
    elif timedateformat == 'datetime':
        return (str(datetime.datetime.now())).split('.')[0]
    elif timedateformat == 'timedate':
        return ((str(
            datetime.datetime.now())).split('.')[0]).split(' ')[1] + ' ' + (
                (str(datetime.datetime.now())).split('.')[0]).split(' ')[0]
    else:
        raise ValueError("Invalid time date format used.")