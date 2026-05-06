def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    try:
        return {
            'authority': 'ISO',
            'namespace': '15924',
            'identifier': name,
            'domain': 'ISO Script Types',
            'display_name': ISO_SCRIPT_TYPES[name] + ' Script Type',
            'display_label': ISO_SCRIPT_TYPES[name],
            'description': ('The display text script type for the ' +
                            ISO_SCRIPT_TYPES[name] + ' script.')
        }
    except KeyError:
        raise NotFound('Script Type:' + name)