def fatal(*args, **kwargs):
    """Log an error message and exit.

    Following arguments are keyword-only.

    :param exitcode: Optional exit code to use
    :param cause: Optional Invoke's Result object, i.e.
                  result of a subprocess invocation
    """
    # determine the exitcode to return to the operating system
    exitcode = None
    if 'exitcode' in kwargs:
        exitcode = kwargs.pop('exitcode')
    if 'cause' in kwargs:
        cause = kwargs.pop('cause')
        if not isinstance(cause, Result):
            raise TypeError(
                "invalid cause of fatal error: expected %r, got %r" % (
                    Result, type(cause)))
        exitcode = exitcode or cause.return_code

    logging.error(*args, **kwargs)
    raise Exit(exitcode or -1)