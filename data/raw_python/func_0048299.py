def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    if name in ISO_LANGUAGE_CODES:
        name = ISO_LANGUAGE_CODES[name]
    if name in ISO_MAJOR_LANGUAGE_TYPES:
        namespace = '639-2'
        lang_name = ISO_MAJOR_LANGUAGE_TYPES[name]
    elif name in ISO_OTHER_LANGUAGE_TYPES:
        namespace = '639-3'
        lang_name = ISO_OTHER_LANGUAGE_TYPES[name]
    else:
        raise NotFound('Language Type: ' + name)

    return {
        'authority': 'ISO',
        'namespace': namespace,
        'identifier': name,
        'domain': 'DisplayText Languages',
        'display_name': lang_name + ' Language Type',
        'display_label': lang_name,
        'description': ('The display text language type for the ' +
                        lang_name + ' language.')
    }