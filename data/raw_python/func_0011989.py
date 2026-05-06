def pop_first_arg(argv):
    """
    find first positional arg (does not start with -), take it out of array and return it separately
    returns (arg, array)
    """
    for arg in argv:
        if not arg.startswith('-'):
            argv.remove(arg)
            return (arg, argv)

    return (None, argv)