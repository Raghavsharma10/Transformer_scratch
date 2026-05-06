def rget(d, key):
    """Recursively get keys from dict, for example:
    'a.b.c' --> d['a']['b']['c'], return None if not exist.
    """
    if not isinstance(d, dict):
        return None
    assert isinstance(key, str) or isinstance(key, list)

    keys = key.split('.') if isinstance(key, str) else key
    cdrs = cdr(keys)
    cars = car(keys)
    return rget(d.get(cars), cdrs) if cdrs else d.get(cars)