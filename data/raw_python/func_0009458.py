def _getFromDate(l, date):
    '''
    returns the index of given or best fitting date
    '''
    try:
        date = _toDate(date)
        i = _insertDateIndex(date, l) - 1
        if i == -1:
            return l[0]
        return l[i]
    except (ValueError, TypeError):
        # ValueError: date invalid / TypeError: date = None
        return l[0]