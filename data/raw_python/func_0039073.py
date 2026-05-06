def fromJson(struct, attributes=None):
    "Convert a JSON struct to a Geometry based on its structure"
    if isinstance(struct, basestring):
        struct = json.loads(struct)
    indicative_attributes = {
        'x': Point,
        'wkid': SpatialReference,
        'paths': Polyline,
        'rings': Polygon,
        'points': Multipoint,
        'xmin': Envelope
    }
    # bbox string
    if isinstance(struct, basestring) and len(struct.split(',')) == 4:
        return Envelope(*map(float, struct.split(',')))
    # Look for telltale attributes in the dict
    if isinstance(struct, dict):
        for key, cls in indicative_attributes.iteritems():
            if key in struct:
                ret = cls.fromJson(dict((str(key), value)
                                   for (key, value) in struct.iteritems()))
                if attributes:
                    ret.attributes = dict((str(key.lower()), val) 
                                           for (key, val)
                                           in attributes.iteritems())
                return ret
    raise ValueError("Unconvertible to geometry")