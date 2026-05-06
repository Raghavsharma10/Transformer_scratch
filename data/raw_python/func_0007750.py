def round_geom(geom, precision=None):
    """Round coordinates of a geometric object to given precision."""
    if geom['type'] == 'Point':
        x, y = geom['coordinates']
        xp, yp = [x], [y]
        if precision is not None:
            xp = [round(v, precision) for v in xp]
            yp = [round(v, precision) for v in yp]
        new_coords = tuple(zip(xp, yp))[0]
    if geom['type'] in ['LineString', 'MultiPoint']:
        xp, yp = zip(*geom['coordinates'])
        if precision is not None:
            xp = [round(v, precision) for v in xp]
            yp = [round(v, precision) for v in yp]
        new_coords = tuple(zip(xp, yp))
    elif geom['type'] in ['Polygon', 'MultiLineString']:
        new_coords = []
        for piece in geom['coordinates']:
            xp, yp = zip(*piece)
            if precision is not None:
                xp = [round(v, precision) for v in xp]
                yp = [round(v, precision) for v in yp]
            new_coords.append(tuple(zip(xp, yp)))
    elif geom['type'] == 'MultiPolygon':
        parts = geom['coordinates']
        new_coords = []
        for part in parts:
            inner_coords = []
            for ring in part:
                xp, yp = zip(*ring)
                if precision is not None:
                    xp = [round(v, precision) for v in xp]
                    yp = [round(v, precision) for v in yp]
                inner_coords.append(tuple(zip(xp, yp)))
            new_coords.append(inner_coords)
    return {'type': geom['type'], 'coordinates': new_coords}