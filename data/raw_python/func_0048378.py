def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    if name in CELESTIAL_COORDINATE_TYPES:
        domain = 'Celestial Coordinate Systems'
        coordinate_name = CELESTIAL_COORDINATE_TYPES[name]
    elif name in GEOGRAPHIC_COORDINATE_TYPES:
        domain = 'Geographic Coordinate Systems'
        coordinate_name = GEOGRAPHIC_COORDINATE_TYPES[name]
    else:
        raise NotFound('Coordinate Type' + name)

    return {
        'authority': 'okapia.net',
        'namespace': 'coordinate',
        'identifier': name,
        'domain': domain,
        'display_name': coordinate_name + ' Type',
        'display_label': coordinate_name,
        'description': ('The type for the ' + coordinate_name + ' System.')
    }