def parse_struct(type_name, body, name):
    '''Returns a :attr:`StructSpec` instance from the input.
    '''
    type_name, name = type_name.strip(), name.strip()
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    members = []

    for line in lines:
        m = match(variable_pat, line)
        if m is None:
            raise Exception('Cannot parse "{}" for "{}"'.format(line, name))

        tp, star, nm, count = [v.strip() if v else '' for v in m.groups()]
        members.append(VariableSpec(tp, star, nm, count))
    return StructSpec(type_name, [n.strip() for n in name.split(',')], members)