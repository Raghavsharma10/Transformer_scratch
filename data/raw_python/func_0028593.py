def slice_repr(slice_instance):
    """
    Turn things like `slice(None, 2, -1)` into `:2:-1`.

    """
    if not isinstance(slice_instance, slice):
        raise TypeError('Unhandled type {}'.format(type(slice_instance)))
    start = slice_instance.start or ''
    stop = slice_instance.stop or ''
    step = slice_instance.step or ''

    msg = '{}:'.format(start)
    if stop:
        msg += '{}'.format(stop)
        if step:
            msg += ':'
    if step:
        msg += '{}'.format(step)
    return msg