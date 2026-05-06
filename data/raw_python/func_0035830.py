def _value_equals(value1, value2, all_close):
    '''
    Get whether 2 values are equal

    value1, value2 : ~typing.Any
    all_close : bool
        compare with np.isclose instead of ==
    '''
    if value1 is None:
        value1 = np.nan
    if value2 is None:
        value2 = np.nan

    are_floats = np.can_cast(type(value1), float) and np.can_cast(type(value2), float)
    if all_close and are_floats:
        return np.isclose(value1, value2, equal_nan=True)
    else:
        if are_floats:
            return value1 == value2 or (value1 != value1 and value2 != value2)
        else:
            return value1 == value2