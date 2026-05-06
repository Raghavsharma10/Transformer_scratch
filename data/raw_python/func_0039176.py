def compare_values(values0, values1):
    """Compares all the values of a single registry key."""
    values0 = {v[0]: v[1:] for v in values0}
    values1 = {v[0]: v[1:] for v in values1}

    created = [(k, v[0], v[1]) for k, v in values1.items() if k not in values0]
    deleted = [(k, v[0], v[1]) for k, v in values0.items() if k not in values1]
    modified = [(k, v[0], v[1]) for k, v in values0.items()
                if v != values1.get(k, None)]

    return created, deleted, modified