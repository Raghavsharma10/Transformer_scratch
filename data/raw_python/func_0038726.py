def get_current_traceback(show_hidden_frames=False, skip=0, context=None, exc_info=None):
    """Get the current exception info as `Traceback` object.  Per default
    calling this method will reraise system exceptions such as generator exit,
    system exit or others.  This behavior can be disabled by passing `False`
    to the function as first parameter.
    """
    if exc_info is None:
        exc_info = sys.exc_info()

    exc_type, exc_value, tb = exc_info
    for x in range(skip):
        if tb.tb_next is None:
            break
        tb = tb.tb_next
    tb = Traceback(exc_type, exc_value, tb, context=context)
    if not show_hidden_frames:
        tb.filter_hidden_frames()
    return tb