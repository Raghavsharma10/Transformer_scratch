def find_prototypes(code):
    """
    Return a list of signatures for each function prototype declared in *code*.
    Format is [(name, [args], rtype), ...].
    """

    prots = []
    lines = code.split('\n')
    for line in lines:
        m = re.match("\s*" + re_func_prot, line)
        if m is not None:
            rtype, name, args = m.groups()[:3]
            if args == 'void' or args.strip() == '':
                args = []
            else:
                args = [tuple(arg.strip().split(' '))
                        for arg in args.split(',')]
            prots.append((name, args, rtype))

    return prots