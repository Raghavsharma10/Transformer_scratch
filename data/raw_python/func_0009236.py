def _float_copy_to_out(out, origin):
    """
    Copy origin to out and return it.

    If ``out`` is None, a new copy (casted to floating point) is used. If
    ``out`` and ``origin`` are the same, we simply return it. Otherwise we
    copy the values.

    """
    if out is None:
        out = origin / 1  # The division forces cast to a floating point type
    elif out is not origin:
        np.copyto(out, origin)
    return out