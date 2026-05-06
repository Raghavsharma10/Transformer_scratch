def _get_pandasindex():
    """
    >>> from hydpy import pub
    >>> pub.timegrids = '2004.01.01', '2005.01.01', '1d'
    >>> from hydpy.core.devicetools import _get_pandasindex
    >>> _get_pandasindex()   # doctest: +ELLIPSIS
    DatetimeIndex(['2004-01-01 12:00:00', '2004-01-02 12:00:00',
    ...
                   '2004-12-30 12:00:00', '2004-12-31 12:00:00'],
                  dtype='datetime64[ns]', length=366, freq=None)
    """
    tg = hydpy.pub.timegrids.init
    shift = tg.stepsize / 2
    index = pandas.date_range(
        (tg.firstdate + shift).datetime,
        (tg.lastdate - shift).datetime,
        (tg.lastdate - tg.firstdate - tg.stepsize) / tg.stepsize + 1)
    return index