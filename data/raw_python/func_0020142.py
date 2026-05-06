def dategenerator(start, end, step=1, desc=False):
    '''Generates dates between *atrt* and *end*.'''
    delta = timedelta(abs(step))
    end = max(start, end)
    if desc:
        dt = end
        while dt >= start:
            yield dt
            dt -= delta
    else:
        dt = start
        while dt <= end:
            yield dt
            dt += delta