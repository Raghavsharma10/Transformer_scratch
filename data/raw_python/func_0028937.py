def _dict_to_pio(d, class_=None):
    """
    Convert a single dictionary object to a Physical Information Object.

    :param d: Dictionary to convert.
    :param class_: Subclass of :class:`.Pio` to produce, if not unambiguous
    :return: Single object derived from :class:`.Pio`.
    """
    d = keys_to_snake_case(d)
    if class_:
        return class_(**d)
    if 'category' not in d:
        raise ValueError('Dictionary does not contains a category field: ' + ', '.join(d.keys()))
    elif d['category'] == 'system':
        return System(**d)
    elif d['category'] == 'system.chemical':
        return ChemicalSystem(**d)
    elif d['category'] == 'system.chemical.alloy':  # Legacy support
        return Alloy(**d)
    elif d['category'] == 'system.chemical.alloy.phase':  # Legacy support
        return ChemicalSystem(**d)
    raise ValueError('Dictionary does not contain a valid top-level category: ' + str(d['category']))