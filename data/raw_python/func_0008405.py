def compile(pattern, *args, **kwargs):
    """ Returns a Pattern from the given string or regular expression.
        Recently compiled patterns are kept in cache
        (if they do not use taxonomies, which are mutable dicts).
    """
    id, p = repr(pattern) + repr(args), pattern
    if id in _cache and not kwargs:
        return _cache[id]
    if isinstance(pattern, basestring):
        p = Pattern.fromstring(pattern, *args, **kwargs)
    if isinstance(pattern, regexp):
        p = Pattern([Constraint(words=[pattern], taxonomy=kwargs.get("taxonomy", TAXONOMY))], *args, **kwargs)
    if len(_cache) > _CACHE_SIZE:
        _cache.clear()
    if isinstance(p, Pattern) and not kwargs:
        _cache[id] = p
    if isinstance(p, Pattern):
        return p
    else:
        raise TypeError("can't compile '%s' object" % pattern.__class__.__name__)