def parse_function_signature(code):
    """
    Return the name, arguments, and return type of the first function
    definition found in *code*. Arguments are returned as [(type, name), ...].
    """
    m = re.search("^\s*" + re_func_decl + "\s*{", code, re.M)
    if m is None:
        print(code)
        raise Exception("Failed to parse function signature. "
                        "Full code is printed above.")
    rtype, name, args = m.groups()[:3]
    if args == 'void' or args.strip() == '':
        args = []
    else:
        args = [tuple(arg.strip().split(' ')) for arg in args.split(',')]
    return name, args, rtype