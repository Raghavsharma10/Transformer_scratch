def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    try:
        return {
            'authority': 'okapia.net',
            'namespace': 'string match types',
            'identifier': name,
            'domain': 'String Match Types',
            'display_name': STRING_MATCH_TYPES[name] + ' String Match Type',
            'display_label': STRING_MATCH_TYPES[name],
            'description': ('The string match type for the ' +
                            STRING_MATCH_TYPES[name])
        }
    except KeyError:
        raise NotFound('String Type: ' + name)