def zscore(ts, **kwargs):
    '''Rolling Z-Score statistics.
    The Z-score is more formally known as ``standardised residuals``.
    To calculate the standardised residuals of a data set,
    the average value and the standard deviation of the data value
    have to be estimated.

    .. math::

        z = \frac{x - \mu(x)}{\sigma(x)}
    '''
    m = ts.rollmean(**kwargs)
    s = ts.rollstddev(**kwargs)
    result = (ts - m)/s
    name = kwargs.get('name', None)
    if name:
        result.name = name
    return result