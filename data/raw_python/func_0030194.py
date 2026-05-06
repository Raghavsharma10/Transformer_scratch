def robust_int(v):
    """Parse an int robustly, ignoring commas and other cruft. """

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return int(v)

    v = str(v).replace(',', '')

    if not v:
        return None

    return int(v)