def parse_attr_signature(sig):
    """ Parse an attribute signature """
    match = ATTR_SIG_RE.match(sig.strip())
    if not match:
        raise RuntimeError('Attribute signature invalid, got ' + sig)
    name, _, params = match.groups()
    if params is not None and params.strip() != '':
        params = split_sig(params)
        params = [parse_param_signature(x) for x in params]
    else:
        params = []
    return (name, params)