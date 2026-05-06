def auto_instantiate(*classes):
    """Creates a decorator that will instantiate objects based on function
    parameter annotations.

    The decorator will check every argument passed into ``f``. If ``f`` has an
    annotation for the specified parameter and the annotation is found in
    ``classes``, the parameter value passed in will be used to construct a new
    instance of the expression that is the annotation.

    An example (Python 3):

    .. code-block:: python

        @auto_instantiate(int)
        def foo(a: int, b: float):
            pass

    Any value passed in as ``b`` is left unchanged. Anything passed as the
    parameter for ``a`` will be converted to :class:`int` before calling the
    function.

    Since Python 2 does not support annotations, the
    :func:`~data.decorators.annotate` function should can be used:

    .. code-block:: python

        @auto_instantiate(int)
        @annotate(a=int)
        def foo(a, b):
            pass


    :param classes: Any number of classes/callables for which
                    auto-instantiation should be performed. If empty, perform
                    for all.

    :note: When dealing with data, it is almost always more convenient to use
           the :func:`~data.decorators.data` decorator instead.
    """
    def decorator(f):
        # collect our argspec
        sig = signature(f)

        @wraps(f)
        def _(*args, **kwargs):
            bvals = sig.bind(*args, **kwargs)

            # replace with instance if desired
            for varname, val in bvals.arguments.items():
                anno = sig.parameters[varname].annotation

                if anno in classes or (len(classes) == 0 and anno != _empty):
                    bvals.arguments[varname] = anno(val)

            return f(*bvals.args, **bvals.kwargs)

        # create another layer by wrapping in a FunctionMaker. this is done
        # to preserve the original signature
        return FunctionMaker.create(
            f, 'return _(%(signature)s)', dict(_=_, __wrapped__=f)
        )

    return decorator