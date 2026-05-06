def tsiterator(ts, dateconverter=None, desc=None,
               clean=False, start_value=None, **kwargs):
    '''An iterator of timeseries as tuples.'''
    dateconverter = dateconverter or default_converter
    yield ['Date'] + ts.names()
    if clean == 'full':
        for dt, value in full_clean(ts, dateconverter, desc, start_value):
             yield (dt,) + tuple(value)
    else:
        if clean:
            ts = ts.clean()
        for dt, value in ts.items(desc=desc, start_value=start_value):
            dt = dateconverter(dt)
            yield (dt,) + tuple(value)