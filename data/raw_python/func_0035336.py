def fail_print(error):
    """Print an error in red text.
    Parameters
        error (HTTPError)
            Error object to print.
    """
    print(COLORS.fail, error.message, COLORS.end)
    print(COLORS.fail, error.errors, COLORS.end)