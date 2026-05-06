def bucket(things, key):
    """Return a map of key -> list of things."""
    ret = defaultdict(list)
    for thing in things:
        ret[key(thing)].append(thing)
    return ret