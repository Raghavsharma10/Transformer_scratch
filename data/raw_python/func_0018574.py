def parse_method_signature(sig):
    """ Parse a method signature of the form: modifier* type name (params) """
    match = METH_SIG_RE.match(sig.strip())
    if not match:
        raise RuntimeError('Method signature invalid: ' + sig)
    modifiers, return_type, name, generic_types, params = match.groups()
    if params.strip() != '':
        params = split_sig(params)
        params = [parse_param_signature(x) for x in params]
    else:
        params = []
    return (modifiers.split(), return_type, name, generic_types, params)