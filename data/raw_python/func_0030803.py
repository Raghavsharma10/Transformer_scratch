def parse_int(v, header_d):
    """Parse as an integer, or a subclass of Int."""

    v = nullify(v)

    if v is None:
        return None

    try:
        # The converson to float allows converting float strings to ints.
        # The conversion int('2.134') will fail.
        return int(round(float(v), 0))
    except (TypeError, ValueError) as e:
        raise CastingError(int, header_d, v, 'Failed to cast to integer')