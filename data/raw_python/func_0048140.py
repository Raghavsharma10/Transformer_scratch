def _parse_unit(unit, placement_str):
    """Parse a unit as part of the unit placement.

    Return the unit as an integer or None.
    Raise a ValueError if the unit is specified but it is not a digit.
    """
    if not unit:
        return None
    try:
        return int(unit)
    except (TypeError, ValueError):
        msg = 'unit in placement {} must be digit'.format(placement_str)
        raise ValueError(msg.encode('utf-8'))