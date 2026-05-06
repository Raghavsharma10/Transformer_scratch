def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    try:
        return {
            'authority': 'okapia.net',
            'namespace': 'TextFormats',
            'identifier': name,
            'domain': 'DisplayText Formats',
            'display_name': FORMAT_TYPES[name] + ' Format Type',
            'display_label': FORMAT_TYPES[name],
            'description': ('The display text format type for the ' +
                            FORMAT_TYPES[name] + ' format.')
        }
    except KeyError:
        raise NotFound('Format Type:' + name)