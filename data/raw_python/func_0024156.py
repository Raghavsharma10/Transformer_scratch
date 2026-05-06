def skip(mapping):
    """
    :param mapping: generator
    :return: filtered generator
    """
    found = set()
    for m in mapping:
        matched_atoms = set(m.values())
        if found.intersection(matched_atoms):
            continue
        found.update(matched_atoms)
        yield m