def parse_param_signature(sig):
    """ Parse a parameter signature of the form: type name (= default)? """
    match = PARAM_SIG_RE.match(sig.strip())
    if not match:
        raise RuntimeError('Parameter signature invalid, got ' + sig)
    groups = match.groups()
    modifiers = groups[0].split()
    typ, name, _, default = groups[-4:]
    return ParamTuple(name=name, typ=typ,
                      default=default, modifiers=modifiers)