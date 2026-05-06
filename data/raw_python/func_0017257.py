def parse(timeseries_expression, method=None, functions=None, debug=False):
    '''Function for parsing :ref:`timeseries expressions <dsl-script>`.
    If succesful, it returns an instance of :class:`dynts.dsl.Expr` which
    can be used to to populate timeseries or scatters once data is available.

    Parsing is implemented using the ply_ module,
    an implementation of lex and yacc parsing tools for Python.

    :parameter expression: A :ref:`timeseries expressions <dsl-script>` string.
    :parameter method: Not yet used.
    :parameter functions: dictionary of functions to use when parsing.
        If not provided the :data:`dynts.function_registry`
        will be used.

        Default ``None``.
    :parameter debug: debug flag for ply_.  Default ``False``.

    For examples and usage check the :ref:`dsl documentation <dsl>`.

    .. _ply: http://www.dabeaz.com/ply/
    '''
    if not parsefunc:
        raise ExpressionError('Could not parse. No parser installed.')
    functions = functions if functions is not None else function_registry
    expr_str = str(timeseries_expression).lower()
    return parsefunc(expr_str, functions, method, debug)