def zoneheight(idf, zonename, debug=False):
    """zone height"""
    zone = idf.getobject('ZONE', zonename)
    surfs = idf.idfobjects['BuildingSurface:Detailed'.upper()]
    zone_surfs = [s for s in surfs if s.Zone_Name == zone.Name]
    floors = [s for s in zone_surfs if s.Surface_Type.upper() == 'FLOOR']
    roofs = [s for s in zone_surfs if s.Surface_Type.upper() == 'ROOF']
    if floors == [] or roofs == []:
        height = zone_height_min2max(idf, zonename)
    else:
        height = zone_floor2roofheight(idf, zonename)
    return height