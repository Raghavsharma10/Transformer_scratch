def invert_dictset(d):
    """Invert a dictionary with keys matching a set of values, turned into lists."""
    # Based on recipe from ASPN
    result = {}
    for k, c in d.items():
        for v in c:
            keys = result.setdefault(v, [])
            keys.append(k)
    return result