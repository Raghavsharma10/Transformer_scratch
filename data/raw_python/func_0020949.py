def parse_date(dateString):
    '''Parses a variety of date formats into a 9-tuple in GMT'''
    if not dateString:
        return None
    for handler in _date_handlers:
        try:
            date9tuple = handler(dateString)
        except (KeyError, OverflowError, ValueError):
            continue
        if not date9tuple:
            continue
        if len(date9tuple) != 9:
            continue
        return date9tuple
    return None