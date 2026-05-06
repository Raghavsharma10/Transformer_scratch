def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    try:
        return {
            'authority': 'gnu.org',
            'namespace': 'Basic Numeric Formats',
            'identifier': name,
            'domain': 'Numeric Format Types',
            'display_name': GNU_BASIC_NUMERIC_FORMAT_TYPES[name] + ' Numeric Format Type',
            'display_label': GNU_BASIC_NUMERIC_FORMAT_TYPES[name],
            'description': ('The type for the ' +
                            GNU_BASIC_NUMERIC_FORMAT_TYPES[name] +
                            ' numeric format.')
        }
    except KeyError:
        raise NotFound('NumericFormat Type: ' + name)