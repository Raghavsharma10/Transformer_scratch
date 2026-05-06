def _match(string, pattern):
    """ Returns True if the pattern matches the given word string.
        The pattern can include a wildcard (*front, back*, *both*, in*side),
        or it can be a compiled regular expression.
    """
    p = pattern
    try:
        if p[:1] == WILDCARD and (p[-1:] == WILDCARD and p[1:-1] in string or string.endswith(p[1:])):
            return True
        if p[-1:] == WILDCARD and not p[-2:-1] == "\\" and string.startswith(p[:-1]):
            return True
        if p == string:
            return True
        if WILDCARD in p[1:-1]:
            p = p.split(WILDCARD)
            return string.startswith(p[0]) and string.endswith(p[-1])
    except:
        # For performance, calling isinstance() last is 10% faster for plain strings.
        if isinstance(p, regexp):
            return p.search(string) is not None
    return False