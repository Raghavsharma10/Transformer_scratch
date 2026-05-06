def mutagen_call(action, path, func, *args, **kwargs):
    """Call a Mutagen function with appropriate error handling.

    `action` is a string describing what the function is trying to do,
    and `path` is the relevant filename. The rest of the arguments
    describe the callable to invoke.

    We require at least Mutagen 1.33, where `IOError` is *never* used,
    neither for internal parsing errors *nor* for ordinary IO error
    conditions such as a bad filename. Mutagen-specific parsing errors and IO
    errors are reraised as `UnreadableFileError`. Other exceptions
    raised inside Mutagen---i.e., bugs---are reraised as `MutagenError`.
    """
    try:
        return func(*args, **kwargs)
    except mutagen.MutagenError as exc:
        log.debug(u'%s failed: %s', action, six.text_type(exc))
        raise UnreadableFileError(path, six.text_type(exc))
    except Exception as exc:
        # Isolate bugs in Mutagen.
        log.debug(u'%s', traceback.format_exc())
        log.error(u'uncaught Mutagen exception in %s: %s', action, exc)
        raise MutagenError(path, exc)