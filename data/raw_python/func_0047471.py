def parse_enum(type_name, body, name):
    '''Returns a :attr:`EnumSpec` instance from the input.
    '''
    type_name, name = type_name.strip(), name.strip()
    lines = [l.strip(' ,') for l in body.splitlines() if l.strip(', ')]
    members = []

    for line in lines:
        vals = [v.strip() for v in line.split('=')]
        if len(vals) == 1:
            members.append(EnumMemberSpec(vals[0], ''))
        else:
            members.append(EnumMemberSpec(*vals))

    return EnumSpec(type_name, [n.strip() for n in name.split(',')], members)