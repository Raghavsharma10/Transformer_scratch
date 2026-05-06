def partLon(ID, chart):
    """ Returns the longitude of an arabic part. """
    # Get diurnal or nocturnal formula
    abc = FORMULAS[ID][0] if chart.isDiurnal() else FORMULAS[ID][1]
    a = objLon(abc[0], chart)
    b = objLon(abc[1], chart)
    c = objLon(abc[2], chart)
    return c + b - a