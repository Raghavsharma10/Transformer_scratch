def combine_xml_points(l, units, handle_units):
    """Combine multiple Point tags into an array."""
    ret = {}
    for item in l:
        for key, value in item.items():
            ret.setdefault(key, []).append(value)

    for key, value in ret.items():
        if key != 'date':
            ret[key] = handle_units(value, units.get(key, None))

    return ret