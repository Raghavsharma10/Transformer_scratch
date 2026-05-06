def find_functions(code):
    """
    Return a list of (name, arguments, return type) for all function 
    definition2 found in *code*. Arguments are returned as [(type, name), ...].
    """
    regex = "^\s*" + re_func_decl + "\s*{"
    
    funcs = []
    while True:
        m = re.search(regex, code, re.M)
        if m is None:
            return funcs
        
        rtype, name, args = m.groups()[:3]
        if args == 'void' or args.strip() == '':
            args = []
        else:
            args = [tuple(arg.strip().split(' ')) for arg in args.split(',')]
        funcs.append((name, args, rtype))
        
        code = code[m.end():]