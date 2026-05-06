def _CaptureException(f, *args, **kwargs):
    """Decorator implementation for capturing exceptions."""
    from ambry.dbexceptions import LoggedException

    b = args[0]  # The 'self' argument

    try:
        return f(*args, **kwargs)
    except Exception as e:
        raise
        try:
            b.set_error_state()
            b.commit()
        except Exception as e2:
            b.log('Failed to set bundle error state: {}'.format(e))
            raise e

        if b.capture_exceptions:
            b.logged_exception(e)
            raise LoggedException(e, b)
        else:
            b.exception(e)
            raise