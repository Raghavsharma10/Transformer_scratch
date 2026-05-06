def find_sources(obj, argspec=None):
    """
    Returns a dictionary of source methods found on this object,
    keyed on method name. Source methods are identified.by argspec,
    a list of argument specifiers. So for e.g. an argpsec of
    :code:`[['self', 'context'], ['s', 'c']]` would match
    methods looking like:

    .. code-block:: python

        def f(self, context):
        ...

    .. code-block:: python

        def f(s, c):
        ...

    is but not

    .. code-block:: python

        def f(self, ctx):
        ...


    """

    if argspec is None:
        argspec = [DEFAULT_ARGSPEC]

    return { n: m for n, m in inspect.getmembers(obj, callable)
        if not n.startswith('_') and
        inspect.getargspec(m).args in argspec }