def parse_indexer_signature(sig):
    """ Parse a indexer signature of the form:
        modifier* type this[params] { (get;)? (set;)? } """
    match = IDXR_SIG_RE.match(sig.strip())
    if not match:
        raise RuntimeError('Indexer signature invalid: ' + sig)
    modifiers, return_type, params, getter, setter = match.groups()
    params = split_sig(params)
    params = [parse_param_signature(x) for x in params]
    return (modifiers.split(), return_type, params,
            getter is not None, setter is not None)