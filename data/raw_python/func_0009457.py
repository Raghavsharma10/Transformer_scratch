def _insertDateIndex(date, l):
    '''
    returns the index to insert the given date in a list
    where each items first value is a date
    '''
    return next((i for i, n in enumerate(l) if n[0] < date), len(l))