def option(*param_decls, **attrs):
    """Attaches an option to the command.  All positional arguments are
    passed as parameter declarations to :class:`Option`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Option` instance manually
    and attaching it to the :attr:`Command.params` list.

    :param cls: the option class to instantiate.  This defaults to
                :class:`Option`.
    """
    def decorator(f):
        if 'help' in attrs:
            attrs['help'] = inspect.cleandoc(attrs['help'])
        OptionClass = attrs.pop('cls', Option)
        _param_memo(f, OptionClass(param_decls, **attrs))
        return f
    return decorator