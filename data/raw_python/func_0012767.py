def zonearea(idf, zonename, debug=False):
    """zone area"""
    zone = idf.getobject('ZONE', zonename)
    surfs = idf.idfobjects['BuildingSurface:Detailed'.upper()]
    zone_surfs = [s for s in surfs if s.Zone_Name == zone.Name]
    floors = [s for s in zone_surfs if s.Surface_Type.upper() == 'FLOOR']
    if debug:
        print(len(floors))
        print([floor.area for floor in floors])
    # area = sum([floor.area for floor in floors])
    if floors != []:
        area = zonearea_floor(idf, zonename)
    else:
        area = zonearea_roofceiling(idf, zonename)
    return area