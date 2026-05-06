def objLon(ID, chart):
    """ Returns the longitude of an object. """
    if ID.startswith('$R'):
        # Return Ruler
        ID = ID[2:]
        obj = chart.get(ID)
        rulerID = essential.ruler(obj.sign)
        ruler = chart.getObject(rulerID)
        return ruler.lon
    elif ID.startswith('Pars'):
        # Return an arabic part
        return partLon(ID, chart)
    else:
        # Return an object
        obj = chart.get(ID)
        return obj.lon