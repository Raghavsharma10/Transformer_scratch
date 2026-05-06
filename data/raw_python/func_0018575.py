def parse_property_signature(sig):
    """ Parse a property signature of the form:
        modifier* type name { (get;)? (set;)? } """
    match = PROP_SIG_RE.match(sig.strip())
    if not match:
        raise RuntimeError('Property signature invalid: ' + sig)
    groups = match.groups()
    if groups[0] is not None:
        modifiers = [x.strip() for x in groups[:-4]]
        groups = groups[-4:]
    else:
        modifiers = []
        groups = groups[1:]
    typ, name, getter, setter = groups
    return (modifiers, typ, name, getter is not None, setter is not None)