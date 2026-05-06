def split_date(value):
    """
        This method splits a date in a tuple.
        value: valid iso date

        ex:
        2016-01-31: ('2016','01','01')
        2016-01: ('2016','01','')
        2016: ('2016','','')
    """
    if not is_valid_date(value):
        return ('', '', '')

    splited = value.split('-')

    try:
        year = splited[0]
    except IndexError:
        year = ''

    try:
        month = splited[1]
    except IndexError:
        month = ''

    try:
        day = splited[2]
    except IndexError:
        day = ''

    return (year, month, day)