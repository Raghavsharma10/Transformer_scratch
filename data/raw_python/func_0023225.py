def _handle_exception(ignore_callback_errors, print_callback_errors, obj,
                      cb_event=None, node=None):
    """Helper for prining errors in callbacks

    See EventEmitter._invoke_callback for a use example.
    """
    if not hasattr(obj, '_vispy_err_registry'):
        obj._vispy_err_registry = {}
    registry = obj._vispy_err_registry

    if cb_event is not None:
        cb, event = cb_event
        exp_type = 'callback'
    else:
        exp_type = 'node'
    type_, value, tb = sys.exc_info()
    tb = tb.tb_next  # Skip *this* frame
    sys.last_type = type_
    sys.last_value = value
    sys.last_traceback = tb
    del tb  # Get rid of it in this namespace
    # Handle
    if not ignore_callback_errors:
        raise
    if print_callback_errors != "never":
        this_print = 'full'
        if print_callback_errors in ('first', 'reminders'):
            # need to check to see if we've hit this yet
            if exp_type == 'callback':
                key = repr(cb) + repr(event)
            else:
                key = repr(node)
            if key in registry:
                registry[key] += 1
                if print_callback_errors == 'first':
                    this_print = None
                else:  # reminders
                    ii = registry[key]
                    # Use logarithmic selection
                    # (1, 2, ..., 10, 20, ..., 100, 200, ...)
                    if ii == (2 ** int(np.log2(ii))):
                        this_print = ii
                    else:
                        this_print = None
            else:
                registry[key] = 1
        if this_print == 'full':
            logger.log_exception()
            if exp_type == 'callback':
                logger.error("Invoking %s for %s" % (cb, event))
            else:  # == 'node':
                logger.error("Drawing node %s" % node)
        elif this_print is not None:
            if exp_type == 'callback':
                logger.error("Invoking %s repeat %s"
                             % (cb, this_print))
            else:  # == 'node':
                logger.error("Drawing node %s repeat %s"
                             % (node, this_print))