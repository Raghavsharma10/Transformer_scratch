def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    try:
        return {
            'authority': 'birdland.mit.edu',
            'namespace': 'unit system',
            'identifier': name,
            'domain': 'Unit System Types',
            'display_name': JEFFS_UNIT_SYSTEM_TYPES[name] + ' Unit System Type',
            'display_label': JEFFS_UNIT_SYSTEM_TYPES[name],
            'description': ('The unit system type for the ' +
                            JEFFS_UNIT_SYSTEM_TYPES[name] + ' System')
        }
    except KeyError:
        raise NotFound('Unit System Type: ' + name)