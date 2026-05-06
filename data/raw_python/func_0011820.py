def merge_dict(data, *args):
    """Merge any number of dictionaries
    """
    results = {}
    for current in (data,) + args:
        results.update(current)
    return results