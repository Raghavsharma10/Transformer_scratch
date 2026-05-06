def print_exception(exception, secret_values=None):
    """
    Prints the exception message and the name of the exception class to stderr.

    :param exception: The exception to print
    :param secret_values: Values to hide in output
    """
    print(brief_exception_text(exception, secret_values), file=sys.stderr)