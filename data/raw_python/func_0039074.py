def fromGeoJson(struct, attributes=None):
    "Convert a GeoJSON-like struct to a Geometry based on its structure"
    if isinstance(struct, basestring):
        struct = json.loads(struct)
    type_map = {
        'Point': Point,
        'MultiLineString': Polyline,
        'LineString': Polyline,
        'Polygon': Polygon,
        'MultiPolygon': Polygon,
        'MultiPoint': Multipoint,
        'Box': Envelope
    }
    if struct['type'] == "Feature":
        return fromGeoJson(struct, struct.get('properties', None))
    elif struct['type'] == "FeatureCollection":
        sr = None
        if 'crs' in struct:
            sr = SpatialReference(struct['crs']['properties']['code'])
            members = map(fromGeoJson, struct['members'])
            for member in members:
                member.spatialReference = sr
            return members
        else:
            return map(fromGeoJson, struct['members'])
    elif struct['type'] in type_map and hasattr(type_map[struct['type']], 
                                              'fromGeoJson'):
        instances = type_map[struct['type']].fromGeoJson(struct)
        i = []
        assert instances is not None, "GeoJson conversion returned a Null geom"
        for instance in instances:
            if 'properties' in struct:
                instance.attributes = struct['properties'].copy()
                if '@esri.sr' in instance.attributes:
                    instance.spatialReference = SpatialReference.fromJson(
                                               instance.attributes['@esri.sr'])
                    del instance.attributes['@esri.sr']
            if attributes:
                if not hasattr(instance, 'attributes'):
                    instance.attributes = {}
                for k, v in attributes.iteritems():
                    instance.attributes[k] = v
            i.append(instance)
        if i:
            if len(i) > 1:
                return i
            return i[0]
    raise ValueError("Unconvertible to geometry")