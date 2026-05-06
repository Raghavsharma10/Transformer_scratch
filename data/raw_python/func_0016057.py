def print_level(log_function, fmt, level, *args):
    """Print a formatted message to stdout prepended by spaces. Useful for
    printing hierarchical information, like bullet lists.

    Note:
        If the application is running in "Silent Mode"
        (i.e., ``_SILENT == True``), this function will return
        immediately and no message will be printed.

    Args:
        log_function: The function that will be called to output the formatted
            message.
        fmt (str): A Python formatted string.
        level (int): Used to determing how many spaces to print. The formula
            is ``'    ' * level ``.
        *args: Variable length list of arguments. Values are plugged into the
            format string.

    Examples:
        >>> print_level("%s %d", 0, "TEST", 0)
        TEST 0
        >>> print_level("%s %d", 1, "TEST", 1)
            TEST 1
        >>> print_level("%s %d", 2, "TEST", 2)
                TEST 2

    """
    if _SILENT:
        return

    msg = fmt % args
    spaces = '    ' * level
    log_function("%s%s" % (spaces, msg))