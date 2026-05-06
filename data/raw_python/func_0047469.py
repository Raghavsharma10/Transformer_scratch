def parse_prototype(prototype):
    '''Returns a :attr:`FunctionSpec` instance from the input.
    '''
    val = ' '.join(prototype.splitlines())
    f = match(func_pat, val)  # match the whole function
    if f is None:
        raise Exception('Cannot parse function prototype "{}"'.format(val))
    ftp, pointer, name, arg = [v.strip() for v in f.groups()]

    args = []
    if arg.strip():  # split each arg into type, zero or more *, and name
        for item in split(arg_split_pat, arg):
            m = match(variable_pat, item.strip())
            if m is None:
                raise Exception('Cannot parse function prototype "{}"'.format(val))

            tp, star, nm, count = [v.strip() if v else '' for v in m.groups()]
            args.append(VariableSpec(tp, star, nm, count))

    return FunctionSpec('FLYCAPTURE2_C_API', ftp, pointer, name, args)