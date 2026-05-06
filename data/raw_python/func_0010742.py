def register(name):
    """Return a decorator that registers the decorated class as a
    resolver with the given *name*."""
    def decorator(class_):
        if name in known_resolvers:
            raise ValueError('duplicate resolver name "%s"' % name)
        known_resolvers[name] = class_
    return decorator