def get_type_data(name):
    """Return dictionary representation of type.

    Can be used to initialize primordium.type.primitives.Type

    """
    name = name.upper()
    if name in CALENDAR_TYPES:
        domain = 'Calendar Types'
        calendar_name = CALENDAR_TYPES[name]
    elif name in ANCIENT_CALENDAR_TYPES:
        domain = 'Ancient Calendar Types'
        calendar_name = ANCIENT_CALENDAR_TYPES[name]
    elif name in ALTERNATE_CALENDAR_TYPES:
        domain = 'Alternative Calendar Types'
        calendar_name = ALTERNATE_CALENDAR_TYPES[name]
    else:
        raise NotFound('Calendar Type: ' + name)

    return {
        'authority': 'okapia.net',
        'namespace': 'calendar',
        'identifier': name,
        'domain': domain,
        'display_name': calendar_name + ' Calendar Type',
        'display_label': calendar_name,
        'description': ('The time type for the ' + calendar_name + ' calendar.')
    }