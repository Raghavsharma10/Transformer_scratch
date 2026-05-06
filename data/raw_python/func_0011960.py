def get_members(app, mod, typ, include_public=None):
    """Return the members of mod of the given type

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param mod: the module with members
    :type mod: module
    :param typ: the typ, ``'class'``, ``'function'``, ``'exception'``, ``'data'``, ``'members'``
    :type typ: str
    :param include_public: list of private members to include to publics
    :type include_public: list | None
    :returns: None
    :rtype: None
    :raises: None
    """
    def include_here(x):
        """Return true if the member should be included in mod.

        A member will be included if it is declared in this module or package.
        If the `jinjaapidoc_include_from_all` option is `True` then the member
        can also be included if it is listed in `__all__`.

        :param x: The member
        :type x: A class, exception, or function.
        :returns: True if the member should be included in mod. False otherwise.
        :rtype: bool
        """
        return (x.__module__ == mod.__name__ or (include_from_all and x.__name__ in all_list))

    all_list = getattr(mod, '__all__', [])
    include_from_all = app.config.jinjaapi_include_from_all

    include_public = include_public or []
    tests = {'class': lambda x: inspect.isclass(x) and not issubclass(x, BaseException) and include_here(x),
             'function': lambda x: inspect.isfunction(x) and include_here(x),
             'exception': lambda x: inspect.isclass(x) and issubclass(x, BaseException) and include_here(x),
             'data': lambda x: not inspect.ismodule(x) and not inspect.isclass(x) and not inspect.isfunction(x),
             'members': lambda x: True}
    items = []
    for name in dir(mod):
        i = getattr(mod, name)
        inspect.ismodule(i)

        if tests.get(typ, lambda x: False)(i):
            items.append(name)
    public = [x for x in items
              if x in include_public or not x.startswith('_')]
    logger.debug('Got members of %s of type %s: public %s and %s', mod, typ, public, items)
    return public, items