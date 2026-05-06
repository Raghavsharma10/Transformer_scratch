def _computeChart(chart, date):
    """ Internal function to return a new chart for
    a specific date using properties from old chart.
    
    """
    pos = chart.pos
    hsys = chart.hsys
    IDs = [obj.id for obj in chart.objects]
    return Chart(date, pos, IDs=IDs, hsys=hsys)