def zone_height_min2max(idf, zonename, debug=False):
    """zone height = max-min"""
    zone = idf.getobject('ZONE', zonename)
    surfs = idf.idfobjects['BuildingSurface:Detailed'.upper()]
    zone_surfs = [s for s in surfs if s.Zone_Name == zone.Name]
    surf_xyzs = [eppy.function_helpers.getcoords(s) for s in zone_surfs]
    surf_xyzs = list(itertools.chain(*surf_xyzs))
    surf_zs = [z for x, y, z in surf_xyzs]
    topz = max(surf_zs)
    botz = min(surf_zs)
    height = topz - botz
    return height