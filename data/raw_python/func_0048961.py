def _validate_constraints(constraints, label, add_error):
    """Validate the given service or machine constraints.

    Use the given label (e.g. "machine X" or "service Y") to describe
    possible errors.
    Use the given add_error callable to register validation error.
    """
    if constraints is None:
        return
    msg = '{} has invalid constraints {}'.format(label, constraints)
    if not isstring(constraints):
        add_error(msg)
        return
    sep = ',' if ',' in constraints else None
    for constraint in constraints.split(sep):
        try:
            key, value = constraint.split('=')
        except (TypeError, ValueError):
            add_error(msg)
            return
        if key not in _CONSTRAINTS:
            add_error(msg)