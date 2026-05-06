def FSkip(params, ctxt, scope, stream, coord):
    """Returns 0 if successful or -1 if the address is out of range
    """
    if len(params) != 1:
        raise errors.InvalidArguments(coord, "{} args".format(len(params)), "FSkip accepts only one argument")

    skip_amt = PYVAL(params[0])
    pos = skip_amt + stream.tell()
    return FSeek([pos], ctxt, scope, stream, coord)