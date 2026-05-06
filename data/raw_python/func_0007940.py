def nextSolarReturn(chart, date):
    """ Returns the solar return of a Chart
    after a specific date.
    
    """
    sun = chart.getObject(const.SUN)
    srDate = ephem.nextSolarReturn(date, sun.lon)
    return _computeChart(chart, srDate)