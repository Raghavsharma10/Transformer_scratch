def printout(*args, **kwargs):
    """
    Print function with extra options for formating text in terminals.
    """

    # TODO(Lukas): conflicts with function names
    color = kwargs.pop('color', {})
    style = kwargs.pop('style', {})
    prefx = kwargs.pop('prefix', '')
    suffx = kwargs.pop('suffix', '')
    ind = kwargs.pop('indent', 0)

    print_args = []
    for arg in args:
        arg = str(arg)
        arg = colorize(arg, **color)
        arg = stylize(arg, **style)
        arg = prefix(arg, prefx)
        arg = indent(arg, ind)
        arg += str(suffx)
        print_args.append(arg)

    print(*print_args, **kwargs)