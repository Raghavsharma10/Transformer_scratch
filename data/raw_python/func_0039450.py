def get_all_attributes(klass_or_instance):
    """Get all attribute members (attribute, property style method).
    """
    pairs = list()
    for attr, value in inspect.getmembers(
            klass_or_instance, lambda x: not inspect.isroutine(x)):
        if not (attr.startswith("__") or attr.endswith("__")):
            pairs.append((attr, value))
    return pairs