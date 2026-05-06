def data(*argnames):
    """Designate an argument as a :class:`~data.Data` argument.

    Works by combining calls to :func:`~data.decorators.auto_instantiate` and
    :func:~data.decorators.annotate` on the named arguments.

    Example:

    .. code-block:: python

       class Foo(object):
           @data('bar')
           def meth(self, foo, bar):
               pass

    Inside ``meth``, ``bar`` will always be a :class:`~data.Data` instance
    constructed from the original value passed as ``bar``.

    :param argnames: List of parameter names that should be data arguments.
    :return: A decorator that converts the named arguments to
             :class:`~data.Data` instances."""
    # make it work if given only one argument (for Python3)
    if len(argnames) == 1 and callable(argnames[0]):
        return data()(argnames[0])

    def decorator(f):
        f = annotate(**dict((argname, Data) for argname in argnames))(f)
        f = auto_instantiate(Data)(f)
        return f
    return decorator