def date_to_timestamp(date):
    """
        date to unix timestamp in milliseconds
    """
    date_tuple = date.timetuple()
    timestamp = calendar.timegm(date_tuple) * 1000
    return timestamp