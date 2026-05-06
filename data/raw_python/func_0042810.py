def _get_best_type_from_mapping(mapping):
    """
    THERE ARE MULTIPLE TYPES IN AN INDEX, PICK THE BEST
    :param mapping: THE ES MAPPING DOCUMENT
    :return: (type_name, mapping) PAIR (mapping.properties WILL HAVE PROPERTIES
    """
    best_type_name = None
    best_mapping = None
    for k, m in mapping.items():
        if k == "_default_":
            continue
        if best_type_name is None or len(m.properties) > len(best_mapping.properties):
            best_type_name = k
            best_mapping = m
    if best_type_name == None:
        return "_default_", mapping["_default_"]
    return best_type_name, best_mapping