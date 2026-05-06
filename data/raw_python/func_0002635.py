def brief_exception_text(exception, secret_values):
    """
    Returns the Exception class and the message of the exception as string.

    :param exception: The exception to format
    :param secret_values: Values to hide in output
    """
    exception_text = _hide_secret_values(str(exception), secret_values)
    return '[{}]\n{}'.format(type(exception).__name__, exception_text)