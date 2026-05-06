def _validate_num_units(num_units, service_name, add_error):
    """Check that the given num_units is valid.

    Use the given service name to describe possible errors.
    Use the given add_error callable to register validation error.

    If no errors are encountered, return the number of units as an integer.
    Return None otherwise.
    """
    if num_units is None:
        # This should be a subordinate charm.
        return 0
    try:
        num_units = int(num_units)
    except (TypeError, ValueError):
        add_error(
            'num_units for service {} must be a digit'.format(service_name))
        return
    if num_units < 0:
        add_error(
            'num_units {} for service {} must be a positive digit'
            ''.format(num_units, service_name))
        return
    return num_units