def parse_type_signature(sig):
    """ Parse a type signature """
    match = TYPE_SIG_RE.match(sig.strip())
    if not match:
        raise RuntimeError('Type signature invalid, got ' + sig)
    groups = match.groups()
    typ = groups[0]
    generic_types = groups[1]
    if not generic_types:
        generic_types = []
    else:
        generic_types = split_sig(generic_types[1:-1])
    is_array = (groups[2] is not None)
    return typ, generic_types, is_array