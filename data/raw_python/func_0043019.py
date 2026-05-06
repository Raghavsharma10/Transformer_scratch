def cibaseglob(pat, base):
    """Case insensitive baseglob.  Note that it's not *actually* case
    insensitive, since glob is insensitive or not based on local semantics.
    Instead, it tries the original version, an upper() version, a lower()
    version, and a swapcase() version of the glob."""
    results = []
    for func in (str, str.upper, str.lower, str.swapcase):
        results += baseglob(func(pat), base)
    return list(sorted(set(results)))