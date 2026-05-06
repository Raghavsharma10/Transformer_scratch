def zonevolume(idf, zonename):
    """zone volume"""
    area = zonearea(idf, zonename)
    height = zoneheight(idf, zonename)
    volume = area * height

    return volume