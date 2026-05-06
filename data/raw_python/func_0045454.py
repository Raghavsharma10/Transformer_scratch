def get_signature(token, contextable=False, comparison=False):
    """
    Gets the signature tuple for any native tag
    contextable searchs for ``as`` variable to update context
    comparison if true uses ``negate`` (p) to ``not`` the result (~p)
    returns (``tag_name``, ``args``, ``kwargs``)
    """
    bits = split(token.contents)
    args, kwargs = (), {}
    if comparison and bits[-1] == 'negate':
        kwargs['negate'] = True
        bits = bits[:-1]
    if contextable and len(bits) > 2 and bits[-2] == 'as':
        kwargs['varname'] = bits[-1]
        bits = bits[:-2]
    kwarg_re = re.compile(r'^([-\w]+)\=(.*)$')
    for bit in bits[1:]:
        match = kwarg_re.match(bit)
        if match:
            kwargs[str(match.group(1))] = force_unicode(match.group(2))
        else:
            args += (bit,)
    return bits[0], args, kwargs