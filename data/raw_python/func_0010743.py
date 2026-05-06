def get_resolver(order=None, options=None, modules=None):
    """Return a location resolver.  The *order* argument, if given,
    should be a list of resolver names; results from resolvers named
    earlier in the list are preferred over later ones.  For a list of
    built-in resolver names, see :doc:`/resolvers`.  The *options*
    argument can be used to pass configuration options to individual
    resolvers, in the form of a dictionary mapping resolver names to
    keyword arguments::

        {'geocode': {'max_distance': 50}}

    The *modules* argument can be used to specify a list of additional
    modules to look for resolvers in.  See :doc:`/develop` for details.
    """
    if not known_resolvers:
        from . import resolvers as carmen_resolvers
        modules = [carmen_resolvers] + (modules or [])
        for module in modules:
            for loader, name, _ in pkgutil.iter_modules(module.__path__):
                full_name = module.__name__ + '.' + name
                loader.find_module(full_name).load_module(full_name)
    if order is None:
        order = ('place', 'geocode', 'profile')
    else:
        order = tuple(order)
    if options is None:
        options = {}
    resolvers = []
    for resolver_name in order:
        if resolver_name not in known_resolvers:
            raise ValueError('unknown resolver name "%s"' % resolver_name)
        resolvers.append((
            resolver_name,
            known_resolvers[resolver_name](**options.get(resolver_name, {}))))
    return ResolverCollection(resolvers)