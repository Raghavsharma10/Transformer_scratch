def parse_header(filename):
    '''Returns a list of :attr:`VariableSpec`, :attr:`FunctionSpec`,
    :attr:`StructSpec`, :attr:`EnumSpec`, :attr:`EnumMemberSpec`, and
    :attr:`TypeDef` instances representing the c header file.
    '''
    with open(filename, 'rb') as fh:
        content = '\n'.join(fh.read().splitlines())

    content = sub('\t', ' ', content)
    content = strip_comments(content)

    # first get the functions
    content = split(func_pat_short, content)

    for i, s in enumerate(content):
        if i % 2 and content[i].strip():  # matched a prototype
            try:
                content[i] = parse_prototype(content[i])
            except Exception as e:
                traceback.print_exc()

    # now process structs
    res = []
    for i, item in enumerate(content):
        if not isinstance(item, str):  # if it's already a func etc. skip it
            res.append(item)
            continue

        items = split(struct_pat, item)
        j = 0
        while j < len(items):
            if not j % 5:
                res.append(items[j])
                j += 1
            else:
                if items[j].strip() == 'enum':
                    res.append(parse_enum(*items[j + 1: j + 4]))
                else:
                    res.append(parse_struct(*items[j + 1: j + 4]))
                j += 4

    # now do remaining simple typedefs
    content = res
    res = []
    for i, item in enumerate(content):
        if not isinstance(item, str):  # if it's already processed skip it
            res.append(item)
            continue

        items = split(typedef_pat, item)
        for j, item in enumerate(items):
            res.append(TypeDef(item.strip()) if j % 2 else item)

    content = [c for c in res if not isinstance(c, str) or c.strip()]
    return content