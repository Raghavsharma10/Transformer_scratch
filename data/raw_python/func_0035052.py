def _create_oracle(oracle_type, **kwargs):
    """A routine for creating a factor oracle."""
    if oracle_type == 'f':
        return FO(**kwargs)
    elif oracle_type == 'a':
        return MO(**kwargs)
    else:
        return MO(**kwargs)