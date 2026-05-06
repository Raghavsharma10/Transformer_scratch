def suppress_stdout():
    """
    Context manager that suppresses stdout.

    Examples:
        >>> with suppress_stdout():
        ...     print('Test print')

        >>> print('test')
        test

    """
    save_stdout = sys.stdout
    sys.stdout = DevNull()
    yield
    sys.stdout = save_stdout