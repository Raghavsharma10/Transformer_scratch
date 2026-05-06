def set_log_level(verbose, match=None, return_old=False):
    """Convenience function for setting the logging level

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
    match : str | None
        String to match. Only those messages that both contain a substring
        that regexp matches ``'match'`` (and the ``verbose`` level) will be
        displayed.
    return_old : bool
        If True, return the old verbosity level and old match.

    Notes
    -----
    If ``verbose=='debug'``, then the ``vispy`` method emitting the log
    message will be prepended to each log message, which is useful for
    debugging. If ``verbose=='debug'`` or ``match is not None``, then a
    small performance overhead is added. Thus it is suggested to only use
    these options when performance is not crucial.

    See also
    --------
    vispy.util.use_log_level
    """
    # This method is responsible for setting properties of the handler and
    # formatter such that proper messages (possibly with the vispy caller
    # prepended) are displayed. Storing log messages is only available
    # via the context handler (use_log_level), so that configuration is
    # done by the context handler itself.
    if isinstance(verbose, bool):
        verbose = 'info' if verbose else 'warning'
    if isinstance(verbose, string_types):
        verbose = verbose.lower()
        if verbose not in logging_types:
            raise ValueError('Invalid argument "%s"' % verbose)
        verbose = logging_types[verbose]
    else:
        raise TypeError('verbose must be a bool or string')
    logger = logging.getLogger('vispy')
    old_verbose = logger.level
    old_match = _lh._vispy_set_match(match)
    logger.setLevel(verbose)
    if verbose <= logging.DEBUG:
        _lf._vispy_set_prepend(True)
    else:
        _lf._vispy_set_prepend(False)
    out = None
    if return_old:
        out = (old_verbose, old_match)
    return out