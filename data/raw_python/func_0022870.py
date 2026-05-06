def _check_valid(key, val, valid):
    """Helper to check valid options"""
    if val not in valid:
        raise ValueError('%s must be one of %s, not "%s"'
                         % (key, valid, val))