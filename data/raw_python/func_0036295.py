def _count(dicts):
    """
    Merge a list of dicts, summing their values.
    """
    counts = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            counts[k] += v
    return counts