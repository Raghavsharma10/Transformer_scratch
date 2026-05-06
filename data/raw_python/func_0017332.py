def prange(ts, **kwargs):
    '''Rolling Percentage range.

    Value between 0 and 1 indicating the position in the rolling range.
    '''
    mi = ts.rollmin(**kwargs)
    ma = ts.rollmax(**kwargs)
    return (ts - mi)/(ma - mi)