def picknthweekday(year, month, dayofweek, hour, minute, whichweek):
    """dayofweek == 0 means Sunday, whichweek 5 means last instance"""
    first = datetime.datetime(year, month, 1, hour, minute)
    weekdayone = first.replace(day=((dayofweek-first.isoweekday())%7+1))
    for n in range(whichweek):
        dt = weekdayone+(whichweek-n)*ONEWEEK
        if dt.month == month:
            return dt