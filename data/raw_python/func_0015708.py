def override(klass):
    """Takes a override class or function and assigns it dunder arguments
    form the overidden one.
    """

    namespace = klass.__module__.rsplit(".", 1)[-1]
    mod_name = const.PREFIX[-1] + "." + namespace
    module = sys.modules[mod_name]

    if isinstance(klass, types.FunctionType):
        def wrap(wrapped):
            setattr(module, klass.__name__, wrapped)
            return wrapped
        return wrap

    old_klass = klass.__mro__[1]
    name = old_klass.__name__
    klass.__name__ = name
    klass.__module__ = old_klass.__module__

    setattr(module, name, klass)

    return klass