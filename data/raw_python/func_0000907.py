def history_date(soup, date_type = None):
    """
    Find a date in the history tag for the specific date_type
    typical date_type values: received, accepted
    """
    if(date_type == None):
        return None

    history_date = raw_parser.history_date(soup, date_type)
    if history_date is None:
        return None
    (day, month, year) = ymd(history_date)
    return date_struct(year, month, day)