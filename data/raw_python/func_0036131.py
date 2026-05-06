def defaults(f, self, *args, **kwargs):
    """
    For ``PARAMETERS`` keys, replace None ``kwargs`` with ``self`` attr values.

    Should be applied on the top of any decorator stack so other decorators see
    the "right" kwargs.

    Will also apply transformations found in ``TRANSFORMS``.
    """
    for name, data in PARAMETERS.iteritems():
        kwargs[name] = kwargs.get(name) or getattr(self, name)
        if 'transform' in data:
            kwargs[name] = data['transform'](kwargs[name])
    return f(self, *args, **kwargs)