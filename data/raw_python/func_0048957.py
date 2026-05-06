def _validate_series(series, label, add_error):
    """Check that the given series is valid.

    Use the given label (e.g. "machine X" or just "bundle") to describe
    possible errors.
    Use the given add_error callable to register validation error.
    """
    if series is None:
        return
    if not isstring(series):
        add_error('{} series must be a string, found {}'.format(label, series))
        return
    if series == 'bundle':
        add_error('{} series must specify a charm series'.format(label))
        return
    if not references.valid_series(series):
        add_error('{} has invalid series {}'.format(label, series))