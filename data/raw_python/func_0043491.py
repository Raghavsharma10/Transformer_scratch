def verbose_print(message, *, operation, verbosity):
    """
    Prints *message* to stderr only if the given *operation* is in the list
    *verbosity*. If "all" is in *verbosity*, all operations are printed.

    **Parameters**

    message : str
        The message to print.
    operation : str
        The type of operation being performed.
    verbosity : [str] or None
        The list of operations to print *message* for. If "all" is contained
        in the list, then all operations are printed. If None, no operation is
        printed.

    **Returns**

    None
    """
    if (verbosity is not None) and ((operation in verbosity) or
                                    ("all"     in verbosity)):
        print(message, file=stderr)