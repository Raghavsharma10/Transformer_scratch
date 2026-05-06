def exception_to_warning(description, category, always_raise=False):
    """
    Catches any exceptions that happen in the corresponding with block
    and instead emits a warning of the given category,
    unless always_raise is True or the environment variable
    OUTDATED_RAISE_EXCEPTION is set to 1, in which caise the exception
    will not be caught.
    """

    try:
        yield
    except Exception:
        # We check for the presence of various globals because we may be seeing the death
        # of the process if this is in a background thread, during which globals
        # get 'cleaned up' and set to None
        if always_raise or os and os.environ and os.environ.get('OUTDATED_RAISE_EXCEPTION') == '1':
            raise

        if warn_with_ignore:
            warn_with_ignore(
                'Failed to %s.\n'
                'Set the environment variable OUTDATED_RAISE_EXCEPTION=1 for a full traceback.'
                % description,
                category,
            )