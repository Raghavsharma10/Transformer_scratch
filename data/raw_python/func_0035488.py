def _checkParam(param, value, paramlimits, paramtypes):
    """Checks if `value` is allowable value for `param`.

    Raises except if `value` is not acceptable, otherwise
    returns `None` if value is acceptable.

    `paramlimits` and `paramtypes` are the `PARAMLIMITS`
    and `PARAMTYPES` attributes of a `Model`.
    """
    assert param in paramlimits, "Invalid param: {0}".format(param)
    (lowlim, highlim) = paramlimits[param]
    paramtype = paramtypes[param]
    if isinstance(paramtype, tuple):
        (paramtype, paramshape) = paramtype
        if not (isinstance(value, paramtype)):
            raise ValueError("{0} must be {1}, not {2}".format(
                    param, paramtype, type(param)))
        if value.shape != paramshape:
            raise ValueError("{0} must have shape {1}, not {2}".format(
                    param, paramshape, value.shape))
        if value.dtype != 'float':
            raise ValueError("{0} must have dtype float, not {1}".format(
                    param, value.dtype))
        if not ((lowlim <= value).all() and (value <= highlim).all()):
            raise ValueError("{0} must be >= {1} and <= {2}, not {3}".format(
                    param, lowlim, highlim, value))
    else:
        if not isinstance(value, paramtype):
            raise ValueError("{0} must be a {1}, not a {2}".format(
                    param, paramtype, type(value)))
        if not (lowlim <= value <= highlim):
            raise ValueError("{0} must be >= {1} and <= {2}, not {3}".format(
                    param, lowlim, highlim, value))